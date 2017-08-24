require "premake-ninja/ninja"

local ROOT = "./"
local SRC = ROOT .. "src/"
local EXTERNAL = ROOT .. "external/"

---------------------------------
-- [ WORKSPACE CONFIGURATION   --
---------------------------------
workspace "deform"                   -- Solution Name
  configurations { "debug", "release" }  -- Optimization/General config mode in VS
  platforms      { "x64" }        -- Dropdown platforms section in VS

  -- _ACTION is the argument you passed into premake5 when you ran it.
  local project_action = "UNDEFINED"
  if _ACTION ~= nill then project_action = _ACTION end

  -- Where the project files (vs project, solution, etc) go
  location( ROOT .. "build")

  flags "FatalWarnings" 
  warnings "Extra"

  defines { "_UNICODE", "UNICODE" }

  linkoptions { "/DEBUG:FULL" }

  -- Debug info for release builds
  filter "configurations:Release"
    buildoptions { "/FS /Zi" }

  -- see 'filter' in the wiki pages
  filter { "configurations:debug" }
    defines { "DEBUG", "_DEBUG", "DF_BUILD_DEBUG" }  
    symbols "On"

  filter { "configurations:release" }
    defines { "NDEBUG", "DF_BUILD_RELEASE" } 
    optimize "Speed"

  filter { "platforms:*32" } architecture "x86"
  filter { "platforms:*64" } architecture "x64"

  -- when building any visual studio project
  filter { "system:windows", "action:vs*"}
    flags { "MultiProcessorCompile", "NoMinimalRebuild" }

  filter "system:windows"
    defines 
    { 
      "DF_PLATFORM_WINDOWS",
      '_CRT_SECURE_NO_WARNINGS',
      '_SCL_SECURE_NO_DEPRECATE', 
    }

  filter {"system:windows", "platforms:*64"}
    defines { "DF_PLATFORM_WIN64" }


  filter {}
  project "deform"
    language "C++"
    targetdir (ROOT .. "bin/%{cfg.platform}/%{cfg.buildcfg}") -- where the output binary goes.
    targetname "Deform" -- the name of the executable saved to targetdir

    kind "ConsoleApp"

    filter {} -- clear filter!

    local src_dir = SRC .. "deform/";
    -- what files the visual studio project/makefile/etc should know about
    files
    { 
      src_dir .. "**.h", 
      src_dir .. "**.hpp", 
      src_dir .. "**.c", 
      src_dir .. "**.cpp",
    }

    vpaths 
    {
      ["Header Files/*"] = { src_dir .. "**.h", src_dir .. "**.hxx", src_dir .. "**.hpp" },
      ["Source Files/*"] = { src_dir .. "**.c", src_dir .. "**.cxx", src_dir .. "**.cpp" },
    }

    includedirs
    {
      src_dir,
      SRC
    }


