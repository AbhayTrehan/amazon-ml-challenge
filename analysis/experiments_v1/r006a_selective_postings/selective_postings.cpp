#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>
#include <omp.h>

struct Term {
  int feature;
  float weight;
  int df;
  double priority;
};

struct ScoredDoc {
  int doc;
  float score;
};

static inline bool score_order(const ScoredDoc &a, const ScoredDoc &b) {
  if (a.score != b.score) return a.score > b.score;
  return a.doc < b.doc;
}

static int selective_postings_impl(
    const int *q_indptr, const int *q_indices, const float *q_data, int n_queries,
    const int *csc_indptr, const int *csc_indices, const float *csc_data,
    const int *a_indptr, const int *a_indices, const float *a_data, int n_docs,
    int posting_budget, int coarse_m, int n_threads, int output_k,
    int *out_indices, float *out_scores, int64_t *out_visits,
    int64_t *out_full_visits, int *out_coarse_counts, int *out_fallback_terms,
    double *timing_seconds) {
  if (posting_budget <= 0 || output_k <= 0 || coarse_m < output_k || n_queries < 0 || n_docs < 0) return 1;
  for (int z = 0; z < 5; ++z) timing_seconds[z] = 0.0;
  omp_set_num_threads(n_threads);

#pragma omp parallel for schedule(dynamic, 8)
  for (int q = 0; q < n_queries; ++q) {
    using Clock = std::chrono::steady_clock;
    double local_t[5] = {0, 0, 0, 0, 0};
    auto t0 = Clock::now();
    std::vector<Term> terms;
    terms.reserve(q_indptr[q + 1] - q_indptr[q]);
    int64_t full = 0;
    for (int p = q_indptr[q]; p < q_indptr[q + 1]; ++p) {
      const int f = q_indices[p];
      const int df = csc_indptr[f + 1] - csc_indptr[f];
      if (df <= 0) continue;
      full += df;
      terms.push_back({f, q_data[p], df, double(q_data[p]) / std::sqrt(double(df))});
    }
    std::sort(terms.begin(), terms.end(), [](const Term &a, const Term &b) {
      if (a.priority != b.priority) return a.priority > b.priority;
      if (a.weight != b.weight) return a.weight > b.weight;
      if (a.df != b.df) return a.df < b.df;
      return a.feature < b.feature;
    });
    auto t1 = Clock::now();
    local_t[0] += std::chrono::duration<double>(t1 - t0).count();

    std::unordered_map<int, float> accum;
    accum.reserve(std::min<int64_t>(std::max(64, coarse_m * 4), std::max<int64_t>(64, posting_budget * 2LL)));
    int64_t visited = 0;
    int consumed = 0;
    auto add_term = [&](const Term &term) {
      const int begin = csc_indptr[term.feature], end = csc_indptr[term.feature + 1];
      for (int p = begin; p < end; ++p)
        accum[csc_indices[p]] += term.weight * csc_data[p];
      visited += end - begin;
      ++consumed;
    };
    t0 = Clock::now();
    while (consumed < int(terms.size()) && (visited < posting_budget || consumed == 0)) add_term(terms[consumed]);
    const int budget_terms = consumed;
    while (consumed < int(terms.size()) && int(accum.size()) < coarse_m) add_term(terms[consumed]);
    t1 = Clock::now();
    local_t[1] += std::chrono::duration<double>(t1 - t0).count();

    t0 = Clock::now();
    std::vector<ScoredDoc> coarse;
    coarse.reserve(accum.size());
    for (const auto &entry : accum) coarse.push_back({entry.first, entry.second});
    const int take = std::min<int>(coarse_m, coarse.size());
    if (take < int(coarse.size())) {
      std::nth_element(coarse.begin(), coarse.begin() + take, coarse.end(), score_order);
      coarse.resize(take);
    }
    std::sort(coarse.begin(), coarse.end(), score_order);
    t1 = Clock::now();
    local_t[2] += std::chrono::duration<double>(t1 - t0).count();

    t0 = Clock::now();
    std::vector<ScoredDoc> exact;
    exact.reserve(coarse.size());
    for (const auto &candidate : coarse) {
      int i = q_indptr[q], iend = q_indptr[q + 1];
      int j = a_indptr[candidate.doc], jend = a_indptr[candidate.doc + 1];
      float dot = 0.0f;
      while (i < iend && j < jend) {
        const int qi = q_indices[i], aj = a_indices[j];
        if (qi == aj) { dot += q_data[i] * a_data[j]; ++i; ++j; }
        else if (qi < aj) ++i;
        else ++j;
      }
      if (dot > 1e-8f) exact.push_back({candidate.doc, dot});
    }
    std::sort(exact.begin(), exact.end(), score_order);
    const int top = std::min<int>(output_k, exact.size());
    for (int k = 0; k < output_k; ++k) {
      const int offset = q * output_k + k;
      out_indices[offset] = k < top ? exact[k].doc : -1;
      out_scores[offset] = k < top ? exact[k].score : 0.0f;
    }
    t1 = Clock::now();
    local_t[3] += std::chrono::duration<double>(t1 - t0).count();

    out_visits[q] = visited;
    out_full_visits[q] = full;
    out_coarse_counts[q] = int(accum.size());
    out_fallback_terms[q] = consumed - budget_terms;
#pragma omp critical
    {
      for (int z = 0; z < 4; ++z) timing_seconds[z] += local_t[z];
    }
  }
  return 0;
}

// Backward-compatible production entry point. Its ABI and K=10 semantics are
// unchanged; the benchmark-only entry point below exposes the already
// exact-reranked pool at a caller-selected depth.
extern "C" int selective_postings(
    const int *q_indptr, const int *q_indices, const float *q_data, int n_queries,
    const int *csc_indptr, const int *csc_indices, const float *csc_data,
    const int *a_indptr, const int *a_indices, const float *a_data, int n_docs,
    int posting_budget, int coarse_m, int n_threads,
    int *out_indices, float *out_scores, int64_t *out_visits,
    int64_t *out_full_visits, int *out_coarse_counts, int *out_fallback_terms,
    double *timing_seconds) {
  return selective_postings_impl(
      q_indptr, q_indices, q_data, n_queries,
      csc_indptr, csc_indices, csc_data,
      a_indptr, a_indices, a_data, n_docs,
      posting_budget, coarse_m, n_threads, 10,
      out_indices, out_scores, out_visits, out_full_visits,
      out_coarse_counts, out_fallback_terms, timing_seconds);
}

extern "C" int selective_postings_topk(
    const int *q_indptr, const int *q_indices, const float *q_data, int n_queries,
    const int *csc_indptr, const int *csc_indices, const float *csc_data,
    const int *a_indptr, const int *a_indices, const float *a_data, int n_docs,
    int posting_budget, int coarse_m, int n_threads, int output_k,
    int *out_indices, float *out_scores, int64_t *out_visits,
    int64_t *out_full_visits, int *out_coarse_counts, int *out_fallback_terms,
    double *timing_seconds) {
  return selective_postings_impl(
      q_indptr, q_indices, q_data, n_queries,
      csc_indptr, csc_indices, csc_data,
      a_indptr, a_indices, a_data, n_docs,
      posting_budget, coarse_m, n_threads, output_k,
      out_indices, out_scores, out_visits, out_full_visits,
      out_coarse_counts, out_fallback_terms, timing_seconds);
}

// Diagnostic-only replay of the identical traversal. This is deliberately
// called outside benchmark timing and reports only whether known truth rows
// were touched and whether they survived partial-score top-M.
extern "C" int selective_truth_diagnostic(
    const int *q_indptr, const int *q_indices, const float *q_data, int n_queries,
    const int *csc_indptr, const int *csc_indices, const float *csc_data,
    const int *truth_indptr, const int *truth_docs,
    int posting_budget, int coarse_m, int n_threads,
    uint8_t *out_exposed, uint8_t *out_coarse) {
  if (posting_budget <= 0 || coarse_m < 10 || n_queries < 0) return 1;
  omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(dynamic, 8)
  for (int q = 0; q < n_queries; ++q) {
    std::vector<Term> terms;
    terms.reserve(q_indptr[q + 1] - q_indptr[q]);
    for (int p = q_indptr[q]; p < q_indptr[q + 1]; ++p) {
      const int f = q_indices[p];
      const int df = csc_indptr[f + 1] - csc_indptr[f];
      if (df > 0) terms.push_back({f, q_data[p], df, double(q_data[p]) / std::sqrt(double(df))});
    }
    std::sort(terms.begin(), terms.end(), [](const Term &a, const Term &b) {
      if (a.priority != b.priority) return a.priority > b.priority;
      if (a.weight != b.weight) return a.weight > b.weight;
      if (a.df != b.df) return a.df < b.df;
      return a.feature < b.feature;
    });
    std::unordered_map<int, float> accum;
    accum.reserve(std::min<int64_t>(std::max(64, coarse_m * 4), std::max<int64_t>(64, posting_budget * 2LL)));
    int64_t visited = 0;
    int consumed = 0;
    auto add_term = [&](const Term &term) {
      const int begin = csc_indptr[term.feature], end = csc_indptr[term.feature + 1];
      for (int p = begin; p < end; ++p)
        accum[csc_indices[p]] += term.weight * csc_data[p];
      visited += end - begin;
      ++consumed;
    };
    while (consumed < int(terms.size()) && (visited < posting_budget || consumed == 0)) add_term(terms[consumed]);
    while (consumed < int(terms.size()) && int(accum.size()) < coarse_m) add_term(terms[consumed]);
    std::vector<ScoredDoc> coarse;
    coarse.reserve(accum.size());
    for (const auto &entry : accum) coarse.push_back({entry.first, entry.second});
    const int take = std::min<int>(coarse_m, coarse.size());
    if (take < int(coarse.size())) {
      std::nth_element(coarse.begin(), coarse.begin() + take, coarse.end(), score_order);
      coarse.resize(take);
    }
    for (int p = truth_indptr[q]; p < truth_indptr[q + 1]; ++p) {
      const int doc = truth_docs[p];
      out_exposed[p] = accum.find(doc) != accum.end();
      uint8_t present = 0;
      for (const auto &candidate : coarse) if (candidate.doc == doc) { present = 1; break; }
      out_coarse[p] = present;
    }
  }
  return 0;
}
