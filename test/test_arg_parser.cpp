#include "catch.hpp"

#include <deform_lib/arg_parser.h>

namespace {
    void init_parser(ArgParser& parser)
    {
        parser.add_positional("command", "some command");
        parser.add_positional("command2", "some command2");
        parser.add_option("named_option", "-n,--named", "named option");
        parser.add_flag("flag_option", "-f,--flag", "flag option");
        parser.add_group();
        parser.add_option("named_option2", "-n2,--named2", "named option");
        parser.add_flag("flag_option2", "-f2,--flag2", "flag option");
        parser.add_group();
        parser.add_option("named_option3", "-n3,--named3", "named option");
        parser.add_flag("flag_option3", "-f3,--flag3", "flag option");
        parser.add_group();
        parser.add_option("array{i}", "-a{i},--array{i}", "array option");
    }
}

TEST_CASE("args", "")
{
    SECTION("succeed")
    {
        int argc = 15;
        char* argv[] = {
            "test.exe",

            "-n", "asd",
            "--flag",

            "--named2", "qwerty",
            "-f2",

            "-a0", "value0",
            "--array1", "value1",
            "-a2", "value2",
            
            "cmd1",
            "cmd2"
        };

        ArgParser parser(argc, argv);
        init_parser(parser);
        parser.print_help();
        REQUIRE(parser.parse());

        REQUIRE(parser.positional("command") == "cmd1");
        REQUIRE(parser.positional("command2") == "cmd2");

        REQUIRE(parser.positional(0) == "test.exe");
        REQUIRE(parser.positional(1) == "cmd1");
        REQUIRE(parser.positional(2) == "cmd2");

        REQUIRE(parser.is_set("named_option"));
        REQUIRE(parser.option("named_option") == "asd");
        REQUIRE(parser.is_set("flag_option"));

        REQUIRE(parser.is_set("named_option2"));
        REQUIRE(parser.option("named_option2") == "qwerty");
        REQUIRE(parser.is_set("flag_option2"));

        REQUIRE(!parser.is_set("named_option3"));
        REQUIRE(!parser.is_set("flag_option3"));

        REQUIRE(parser.is_set("array0"));
        REQUIRE(parser.option("array0") == "value0");
        REQUIRE(parser.is_set("array1"));
        REQUIRE(parser.option("array1") == "value1");
        REQUIRE(parser.is_set("array2"));
        REQUIRE(parser.option("array2") == "value2");
    }
    SECTION("array")
    {
        int argc = 5;
        char* argv[] = {
            "test.exe",
            
            "-a", "value0",
            
            "cmd1",
            "cmd2"
        };
        
        ArgParser parser(argc, argv);
        init_parser(parser);
        REQUIRE(parser.parse());
        REQUIRE(parser.option("array0") == "value0");
    }
    SECTION("unexpected arg")
    {
        int argc = 17;
        char* argv[] = {
            "test.exe",

            "-n", "asd",
            "--flag",

            "--named2", "qwerty",
            "-f2",

            "-a0", "value0",
            "--array1", "value1",
            "-a2", "value2",
            
            "--unexpected", "asd",

            "cmd1",
            "cmd2"
        };

        ArgParser parser(argc, argv);
        init_parser(parser);
        REQUIRE(!parser.parse());
    }
    SECTION("missing positional")
    {
        int argc = 14;
        char* argv[] = {
            "test.exe",

            "-n", "asd",
            "--flag",

            "--named2", "qwerty",
            "-f2",

            "-a0", "value0",
            "--array1", "value1",
            "-a2", "value2",

            "cmd1"
        };

        ArgParser parser(argc, argv);
        init_parser(parser);
        REQUIRE(!parser.parse());
    }
    SECTION("missing value")
    {
        int argc = 5;
        char* argv[] = {
            "test.exe",

            "cmd1",
            "cmd2",
            
            "--named",
            "--flag",
        };
        ArgParser parser(argc, argv);
        init_parser(parser);
        parser.parse();

        REQUIRE(!parser.is_set("flag_option"));
        REQUIRE(parser.option("named_option") == "--flag");
    }
    SECTION("missing value 2")
    {
        int argc = 4;
        char* argv[] = {
            "test.exe",

            "cmd1",
            "cmd2",
            
            "--named",
        };
        ArgParser parser(argc, argv);
        init_parser(parser);
        REQUIRE(!parser.parse());
    }

}
