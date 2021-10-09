# CMake generated Testfile for 
# Source directory: /data/home/nanwang/misc/tools/cmake-3.20.3/Utilities/cmcurl
# Build directory: /data/home/nanwang/misc/tools/cmake-3.20.3/Utilities/cmcurl
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(curl "curltest" "http://open.cdash.org/user.php")
set_tests_properties(curl PROPERTIES  _BACKTRACE_TRIPLES "/data/home/nanwang/misc/tools/cmake-3.20.3/Utilities/cmcurl/CMakeLists.txt;1468;add_test;/data/home/nanwang/misc/tools/cmake-3.20.3/Utilities/cmcurl/CMakeLists.txt;0;")
subdirs("lib")
