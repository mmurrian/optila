#pragma once

#include <type_traits>

namespace optila::details {
struct eager_strategy_tag {};
struct lazy_strategy_tag {};
struct policy_strategy_tag {};

template <typename OnceStrategy, typename ThenStrategy>
struct once_and_then_strategy {};

template <typename Strategy>
struct current_strategy;

template <typename OnceStrategy, typename ThenStrategy>
struct current_strategy<once_and_then_strategy<OnceStrategy, ThenStrategy>> {
  using type = OnceStrategy;
};

template <typename Strategy>
struct current_strategy {
  using type = Strategy;
};

template <typename Strategy>
using current_strategy_t = typename current_strategy<Strategy>::type;

template <typename Strategy>
struct next_strategy;

template <typename OnceStrategy, typename ThenStrategy>
struct next_strategy<once_and_then_strategy<OnceStrategy, ThenStrategy>> {
  using type = ThenStrategy;
};

template <typename Strategy>
struct next_strategy {
  using type = Strategy;
};

template <typename Strategy>
using next_strategy_t = typename next_strategy<Strategy>::type;

template <typename Strategy, typename EvaluatorPolicyChain>
struct use_lazy_strategy;

template <typename Strategy>
struct use_lazy_strategy<Strategy, void>
    : std::is_same<Strategy, lazy_strategy_tag> {};

template <typename Strategy, typename EvaluatorPolicyChain>
struct use_lazy_strategy {
  constexpr static bool value =
      std::is_same_v<current_strategy_t<Strategy>, lazy_strategy_tag> ||
      (std::is_same_v<current_strategy_t<Strategy>, policy_strategy_tag> &&
       EvaluatorPolicyChain::lazy_evaluation);
};

template <typename Strategy, typename EvaluatorPolicyChain>
inline constexpr bool use_lazy_strategy_v =
    use_lazy_strategy<Strategy, EvaluatorPolicyChain>::value;

template <typename Strategy, typename EvaluatorPolicyChain>
struct use_eager_strategy;

template <typename Strategy>
struct use_eager_strategy<Strategy, void>
    : std::is_same<current_strategy_t<Strategy>, eager_strategy_tag> {};

template <typename Strategy, typename EvaluatorPolicyChain>
struct use_eager_strategy {
  constexpr static bool value =
      std::is_same_v<current_strategy_t<Strategy>, eager_strategy_tag> ||
      (std::is_same_v<current_strategy_t<Strategy>, policy_strategy_tag> &&
       !EvaluatorPolicyChain::lazy_evaluation);
};

template <typename Strategy, typename EvaluatorPolicyChain>
inline constexpr bool use_eager_strategy_v =
    use_eager_strategy<Strategy, EvaluatorPolicyChain>::value;

};  // namespace optila::details