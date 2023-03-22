/**
 * @file testobservationtable.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-03-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "observation_table.h"
#include "catch.hpp"
#include "definitions.h"

#include <vector>

using namespace std;
using namespace active_learning_namespace;

//using Catch::Matchers::Equals;

const int ACCEPTING = 1;
const int REJECTING = 0;

TEST_CASE("Observation table: Construction and simple insertions", "[memory]") {

    vector<int> alp = {1, 2, 3};
    observation_table obs = observation_table(alp);
    const auto nullvector = vector<int>{};

    SECTION("Check initialization"){
        CHECK(obs.get_incomplete_rows().size() == alp.size()+1); // plus one for empty string

        const auto upper_table = obs.get_upper_table();
        const auto lower_table = obs.get_lower_table();

        CHECK(upper_table.size() == 0);
        CHECK(lower_table.size() == alp.size()+1);

        const auto& all_columns = obs.get_column_names();
        REQUIRE(all_columns.size()==1);
    }

    SECTION("Check behavior when expanding table"){
        const auto& all_columns = obs.get_column_names();

        CHECK_FALSE(obs.is_closed());
        // at first we do insertions of level 1 strings
        for(const int i: alp){
            obs.insert_record(vector<int>{i}, vector<int>(), ACCEPTING); // all strings are accepting
        }
         for(const int i: alp){
            CHECK(obs.has_record(vector<int>{i}, vector<int>()));
            CHECK(obs.get_answer(vector<int>{i}, vector<int>()) == ACCEPTING);
        }
        //CHECK_THROWS(obs.is_closed()); // macro does not work as intended

        // check the null-string
        CHECK_FALSE(obs.has_record(nullvector, vector<int>()));
        obs.insert_record(nullvector, vector<int>(), REJECTING);
        CHECK(obs.has_record(nullvector, vector<int>()));
        CHECK(obs.get_answer(nullvector, vector<int>()) == REJECTING);

        // strings of level 2
        obs.extend_lower_table();
        for(const int i: alp){
            for(const int j: alp){
                obs.insert_record(vector<int>{i, j}, vector<int>(), REJECTING);
            }
        }

        for(const int i: alp){
            for(const int j: alp){
                CHECK(obs.has_record(vector<int>{i, j}, vector<int>()));
                CHECK(obs.get_answer(vector<int>{i, j}, vector<int>()) == REJECTING);            
            }
        }

        const vector<int> suffix{2, 2, 3};
        obs.extent_columns(suffix);
        const list<pref_suf_t> irows = list<pref_suf_t>(obs.get_incomplete_rows());
        CHECK(irows.size() > 0);

        for(const auto& row: irows){
            obs.mark_row_complete(row);
        }
        REQUIRE(obs.get_incomplete_rows().size()==0);
    }
}