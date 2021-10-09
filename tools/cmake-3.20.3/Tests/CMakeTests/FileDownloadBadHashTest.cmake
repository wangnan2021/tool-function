if(NOT "/data/home/nanwang/misc/tools/cmake-3.20.3/Tests/CMakeTests" MATCHES "^/")
  set(slash /)
endif()
set(url "file://${slash}/data/home/nanwang/misc/tools/cmake-3.20.3/Tests/CMakeTests/FileDownloadInput.png")
set(dir "/data/home/nanwang/misc/tools/cmake-3.20.3/Tests/CMakeTests/downloads")

file(DOWNLOAD
  ${url}
  ${dir}/file3.png
  TIMEOUT 2
  STATUS status
  EXPECTED_HASH SHA1=5555555555555555555555555555555555555555
  )
