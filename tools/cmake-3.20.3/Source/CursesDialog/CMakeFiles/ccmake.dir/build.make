# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /data/home/nanwang/misc/tools/cmake-3.20.3/Bootstrap.cmk/cmake

# The command to remove a file.
RM = /data/home/nanwang/misc/tools/cmake-3.20.3/Bootstrap.cmk/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/home/nanwang/misc/tools/cmake-3.20.3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/home/nanwang/misc/tools/cmake-3.20.3

# Include any dependencies generated for this target.
include Source/CursesDialog/CMakeFiles/ccmake.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/CursesDialog/CMakeFiles/ccmake.dir/progress.make

# Include the compile flags for this target's objects.
include Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make

Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.o: Source/CursesDialog/ccmake.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.o -MF CMakeFiles/ccmake.dir/ccmake.cxx.o.d -o CMakeFiles/ccmake.dir/ccmake.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/ccmake.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/ccmake.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/ccmake.cxx > CMakeFiles/ccmake.dir/ccmake.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/ccmake.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/ccmake.cxx -o CMakeFiles/ccmake.dir/ccmake.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o: Source/CursesDialog/cmCursesBoolWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesBoolWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesBoolWidget.cxx > CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesBoolWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o: Source/CursesDialog/cmCursesCacheEntryComposite.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesCacheEntryComposite.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesCacheEntryComposite.cxx > CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesCacheEntryComposite.cxx -o CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.o: Source/CursesDialog/cmCursesColor.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) -DHAVE_CURSES_USE_DEFAULT_COLORS $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesColor.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesColor.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesColor.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesColor.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) -DHAVE_CURSES_USE_DEFAULT_COLORS $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesColor.cxx > CMakeFiles/ccmake.dir/cmCursesColor.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesColor.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) -DHAVE_CURSES_USE_DEFAULT_COLORS $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesColor.cxx -o CMakeFiles/ccmake.dir/cmCursesColor.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o: Source/CursesDialog/cmCursesDummyWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesDummyWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesDummyWidget.cxx > CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesDummyWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o: Source/CursesDialog/cmCursesFilePathWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesFilePathWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesFilePathWidget.cxx > CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesFilePathWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.o: Source/CursesDialog/cmCursesForm.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesForm.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesForm.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesForm.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesForm.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesForm.cxx > CMakeFiles/ccmake.dir/cmCursesForm.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesForm.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesForm.cxx -o CMakeFiles/ccmake.dir/cmCursesForm.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o: Source/CursesDialog/cmCursesLabelWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesLabelWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesLabelWidget.cxx > CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesLabelWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o: Source/CursesDialog/cmCursesLongMessageForm.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesLongMessageForm.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesLongMessageForm.cxx > CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesLongMessageForm.cxx -o CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o: Source/CursesDialog/cmCursesMainForm.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesMainForm.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesMainForm.cxx > CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesMainForm.cxx -o CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o: Source/CursesDialog/cmCursesOptionsWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesOptionsWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesOptionsWidget.cxx > CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesOptionsWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o: Source/CursesDialog/cmCursesPathWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesPathWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesPathWidget.cxx > CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesPathWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o: Source/CursesDialog/cmCursesStringWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesStringWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesStringWidget.cxx > CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesStringWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.s

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/flags.make
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o: Source/CursesDialog/cmCursesWidget.cxx
Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o: Source/CursesDialog/CMakeFiles/ccmake.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o -MF CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o.d -o CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o -c /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesWidget.cxx

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ccmake.dir/cmCursesWidget.cxx.i"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesWidget.cxx > CMakeFiles/ccmake.dir/cmCursesWidget.cxx.i

Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ccmake.dir/cmCursesWidget.cxx.s"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/cmCursesWidget.cxx -o CMakeFiles/ccmake.dir/cmCursesWidget.cxx.s

# Object files for target ccmake
ccmake_OBJECTS = \
"CMakeFiles/ccmake.dir/ccmake.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesColor.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesForm.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o" \
"CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o"

# External object files for target ccmake
ccmake_EXTERNAL_OBJECTS =

bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/ccmake.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesBoolWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesCacheEntryComposite.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesColor.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesDummyWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesFilePathWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesForm.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLabelWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesLongMessageForm.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesMainForm.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesOptionsWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesPathWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesStringWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/cmCursesWidget.cxx.o
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/build.make
bin/ccmake: Source/libCMakeLib.a
bin/ccmake: Source/CursesDialog/form/libcmForm.a
bin/ccmake: Source/kwsys/libcmsys.a
bin/ccmake: Utilities/std/libcmstd.a
bin/ccmake: Utilities/cmexpat/libcmexpat.a
bin/ccmake: Utilities/cmlibarchive/libarchive/libcmlibarchive.a
bin/ccmake: Utilities/cmliblzma/libcmliblzma.a
bin/ccmake: Utilities/cmzstd/libcmzstd.a
bin/ccmake: Utilities/cmbzip2/libcmbzip2.a
bin/ccmake: Utilities/cmcurl/lib/libcmcurl.a
bin/ccmake: Utilities/cmzlib/libcmzlib.a
bin/ccmake: /usr/lib64/libssl.so
bin/ccmake: /usr/lib64/libcrypto.so
bin/ccmake: Utilities/cmnghttp2/libcmnghttp2.a
bin/ccmake: Utilities/cmjsoncpp/libcmjsoncpp.a
bin/ccmake: Utilities/cmlibuv/libcmlibuv.a
bin/ccmake: Utilities/cmlibrhash/libcmlibrhash.a
bin/ccmake: /usr/lib64/libncurses.so
bin/ccmake: Source/CursesDialog/CMakeFiles/ccmake.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/home/nanwang/misc/tools/cmake-3.20.3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking CXX executable ../../bin/ccmake"
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ccmake.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Source/CursesDialog/CMakeFiles/ccmake.dir/build: bin/ccmake
.PHONY : Source/CursesDialog/CMakeFiles/ccmake.dir/build

Source/CursesDialog/CMakeFiles/ccmake.dir/clean:
	cd /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog && $(CMAKE_COMMAND) -P CMakeFiles/ccmake.dir/cmake_clean.cmake
.PHONY : Source/CursesDialog/CMakeFiles/ccmake.dir/clean

Source/CursesDialog/CMakeFiles/ccmake.dir/depend:
	cd /data/home/nanwang/misc/tools/cmake-3.20.3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/home/nanwang/misc/tools/cmake-3.20.3 /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog /data/home/nanwang/misc/tools/cmake-3.20.3 /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog /data/home/nanwang/misc/tools/cmake-3.20.3/Source/CursesDialog/CMakeFiles/ccmake.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Source/CursesDialog/CMakeFiles/ccmake.dir/depend

