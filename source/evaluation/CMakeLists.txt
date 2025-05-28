#including those directories enables us to use active learning structures in heuristics as well
include_directories(
  "../active_learning/active_learning_util"
  "../active_learning/memory"
  "../active_learning/memory/incomplete_information"
  "../active_learning/system_under_learning"
  "../active_learning/system_under_learning/neural_network_suls"
  "../active_learning/system_under_learning/benchmark_parsers"
)


add_library(Evaluation STATIC
            aic.h
            aic.cpp
            alergia.h
            alergia.cpp
            alergia94.h
            alergia94.cpp
            count_types.h
            count_types.cpp
            css.h
            css.cpp
            edsm.h
            edsm.cpp
            evaluate.h
            evaluate.cpp
            gini.h
            gini.cpp
            kldistance.h
            kldistance.cpp
            likelihood.h
            likelihood.cpp
            lsharp_eval.h
            lsharp_eval.cpp
            lstar_evaluation.h
            lstar_evaluation.cpp
            mealy.h
            mealy.cpp
            membership_state_comparator.h
            membership_state_comparator.cpp
            mse_error.h
            mse_error.cpp
            overlap.h
            overlap.cpp
            overlap4logs.h
            overlap4logs.cpp
            paul_heuristic.h
            paul_heuristic.cpp
            paul_heuristic_2.h
            paul_heuristic_2.cpp
            rti+.h
            rti+.cpp
            space-saving.h
            space-saving.cpp
            probabilistic_heuristics/probabilistic_heuristic_interface.h
            probabilistic_heuristics/probabilistic_heuristic_interface.cpp
            probabilistic_heuristics/string_probability_estimator.h
            probabilistic_heuristics/string_probability_estimator.cpp
            probabilistic_heuristics/weight_comparator.h
            probabilistic_heuristics/weight_comparator.cpp)
if (ENABLE_DATABASE)
    target_link_libraries(Evaluation libpqxx::pqxx)
endif()
