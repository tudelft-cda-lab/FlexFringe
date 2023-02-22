include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/active_learning_util"
        "${CMAKE_CURRENT_SOURCE_DIR}/memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/teacher"
        "${CMAKE_CURRENT_SOURCE_DIR}/system_under_learning"
        "${CMAKE_CURRENT_SOURCE_DIR}/counterexample_strategies"
        "${CMAKE_CURRENT_SOURCE_DIR}/equivalence_oracle"
        "${CMAKE_CURRENT_SOURCE_DIR}/observation_table"
        "${CMAKE_CURRENT_SOURCE_DIR}/algorithms"
        )

add_library(ActiveLearning STATIC

            active_learning_util/definitions.h

            algorithms/lstar.h
            algorithms/lstar.cpp

            equivalence_oracle/eq_oracle_base.h
            equivalence_oracle/eq_oracle_base.cpp
            equivalence_oracle/lstar_oracle.h
            equivalence_oracle/lstar_oracle.cpp

            memory/observation_table.h
            memory/observation_table.cpp

            system_under_learning/sul_base.h
            system_under_learning/sul_base.cpp
            system_under_learning/positive_traces_sul.h
            system_under_learning/positive_traces_sul.cpp

            teacher/teacher_base.h
            teacher/lstar_teacher.h
            teacher/lstar_teacher.cpp
)