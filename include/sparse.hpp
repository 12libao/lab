#pragma once
#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "lapackage.hpp"
#include "utils.hpp"

/**
 * A sparse matrix in COO format
 *
 * Members:
 *   numRows: number of rows, initialized to 0
 *   numCols: number of columns, initialized to 0
 *   rowIndices: row indices of non-zero elements
 *   colIndices: column indices of non-zero elements
 *   values: values of non-zero elements
 */
template <typename T, typename I>
struct COOMatrix {
  I numRows = 0;
  I numCols = 0;
  Kokkos::View<I*> rowIndices;
  Kokkos::View<I*> colIndices;
  Kokkos::View<T*> values;
};

/**
 * A sparse matrix in CSR format
 *
 * Members:
 *   numRows: number of rows, initialized to 0
 *   numCols: number of columns, initialized to 0
 *   rowPtr: row pointers
 *   colIndices: column indices of non-zero elements
 *   values: values of non-zero elements
 *
 * Data Container
 *   Kokkos::View
 */
template <typename T, typename I>
struct CSRMatrix {
  I numRows = 0;
  I numCols = 0;
  Kokkos::View<I*> rowPtr;
  Kokkos::View<I*> colIndices;
  Kokkos::View<T*> values;
};

/**
 * Convert a COO matrix to CSR format
 *
 * Input:
 *   coo: a COO matrix
 *
 * Output:
 *   csr: a CSR matrix
 *
 * Note:
 *   This function not assume that the input COO matrix is sorted.
 *   This function can deal with duplicate entries.
 *
 * Example:
 *
 *   Matrix: [3, 0, 1, 0]
 *           [0, 2, 0, 0]
 *           [0, 0, 0, 0]
 *           [0, 0, 0, 1]
 *
 *   coo.rowIndices = [3, 0, 0, 1, 1, 0, 0]
 *   coo.colIndices = [3, 2, 0, 1, 1, 0, 0]
 *   coo.values = [1, 1, 1, 1, 1, 1, 1]
 *
 *   csr.rowPtr = [0, 2, 3, 3, 4]
 *   csr.colIndices = [0, 2, 1, 3]
 *   csr.values = [3, 1, 2, 1]
 */
template <typename T, typename I>
void cooToCsr(const COOMatrix<T, I>& coo, CSRMatrix<T, I>& csr) {
  // Determine the number of rows and columns
  if (coo.numRows == 0) {
    Kokkos::parallel_reduce(
        "Compute number of rows", coo.values.extent(0),
        KOKKOS_LAMBDA(I n, I & ncols_) { ncols_ = std::max(ncols_, coo.colIndices(n) + 1); },
        Kokkos::Max<I>(csr.numRows));
  } else {
    csr.numRows = coo.numRows;
  }

  // Create a dictionary to store (row, col) pairs as keys and sum Bx as Bx
  std::map<std::pair<I, I>, T> coo_dict;

  // Sum duplicate entries and compute the number of non-zero elements in each row
  Kokkos::View<I*> row_counts("row_counts", csr.numRows);

  for (size_t i = 0; i < coo.values.extent(0); ++i) {
    I row = coo.rowIndices(i);
    I col = coo.colIndices(i);
    std::pair<I, I> key = std::make_pair(row, col);
    if (coo_dict.find(key) != coo_dict.end()) {
      coo_dict[key] += coo.values(i);
    } else {
      coo_dict[key] = coo.values(i);
      row_counts(row) += 1;
    }
  }

  // Compute the Bp array
  Kokkos::View<I*> Bp("Bp", csr.numRows + 1);
  Kokkos::parallel_scan(
      "COO to CSR", csr.numRows, KOKKOS_LAMBDA(I n, I & update, const bool& final) {
        update += row_counts(n);
        if (final) {
          Bp(n + 1) = update;
        }
      });

  // Create a view for the keys
  Kokkos::View<std::pair<I, I>*> keys("keys", coo_dict.size());
  size_t i = 0;
  for (const auto& entry : coo_dict) {
    keys(i) = entry.first;
    ++i;
  }

  // Create views for Bj and Bx
  Kokkos::View<I*> Bj("Bj", keys.extent(0));
  Kokkos::View<T*> Bx("Bx", keys.extent(0));

  // Fill Bj and Bx
  for (size_t i = 0; i < keys.extent(0); ++i) {
    Bj(i) = keys(i).second;
    Bx(i) = coo_dict[keys(i)];
  }

  // Assign the views to the CSR matrix
  csr.rowPtr = Bp;
  csr.colIndices = Bj;
  csr.values = Bx;
}

/**
 * Convert a COO matrix to CSR format
 *
 * input:
 *   Ax: values of non-zero elements
 *   Ai: row indices of non-zero elements
 *   Aj: column indices of non-zero elements
 *
 * returns:
 *   Bx: values of non-zero elements
 *   Bp: row pointers
 *   Bj: column indices of non-zero elements
 *
 * Note:
 *   This function not assume that the input COO matrix is sorted.
 *   This function can deal with duplicate entries.
 */
template <typename T, typename I>
std::tuple<std::vector<T>, std::vector<I>, std::vector<I>> coo_to_csr(const std::vector<T>& Ax,
                                                                      const std::vector<I>& Ai,
                                                                      const std::vector<I>& Aj) {
  // Determine the number of rows and columns
  I num_rows = *std::max_element(Ai.begin(), Ai.end()) + 1;

  // Create a dictionary to store (row, col) pairs as keys and sum Bx as Bx
  std::map<std::pair<I, I>, T> coo_dict;

  // Sum duplicate entries and compute the number of non-zero elements in each row
  std::vector<I> row_counts(num_rows, 0);
  row_counts.reserve(Ax.size());

  for (size_t i = 0; i < Ax.size(); ++i) {
    I row = Ai[i];
    I col = Aj[i];
    std::pair<I, I> key = std::make_pair(row, col);
    if (coo_dict.find(key) != coo_dict.end()) {
      coo_dict[key] += Ax[i];
    } else {
      coo_dict[key] = Ax[i];
      row_counts[row] += 1;
    }
  }

  // Compute the Bp array
  std::vector<I> Bp(num_rows + 1, 0);
  I cumulative_sum = 0;
  for (I i = 0; i < num_rows; ++i) {
    cumulative_sum += row_counts[i];
    Bp[i + 1] = cumulative_sum;
  }

  std::vector<std::pair<I, I>> keys;
  keys.reserve(coo_dict.size());
  for (const auto& entry : coo_dict) {
    keys.push_back(entry.first);
  }

  std::vector<I> Bj;
  std::vector<T> Bx;
  Bj.reserve(keys.size());
  Bx.reserve(keys.size());

  for (const auto& key : keys) {
    Bj.emplace_back(key.second);
    Bx.emplace_back(coo_dict[key]);
  }

  return std::make_tuple(Bx, Bp, Bj);
}

#endif  // SPARSE_HPP