@echo off

setlocal

rem Assume env is already setup if VisualStudioVersion == 15.0
if "%VisualStudioVersion%"=="15.0" ( goto :build )

setlocal 
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64

:build
call tools\ninja.exe -C build %1