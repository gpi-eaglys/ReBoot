#include <gtest/gtest.h>

#include <utils.h>

TEST(Scaffold, SetAndGetNumThreads) {
    set_num_threads(4);
    EXPECT_EQ(get_num_threads(), 4);
}
