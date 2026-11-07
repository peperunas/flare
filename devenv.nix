{ pkgs, lib, ... }:

let
  cuda = pkgs.cudaPackages_13_2;
in
{
  packages = with pkgs; [
    cargo
    cmake
    git
    liburing
    ninja
    openssl
    perl
    pkg-config
    protobuf
    python312
    rustc
    stdenv.cc
    uv
    zlib
  ] ++ [
    cuda.cudatoolkit
  ];

  env = {
    CUDA_HOME = "${cuda.cudatoolkit}";
    CUDA_PATH = "${cuda.cudatoolkit}";
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      cuda.cudatoolkit
      "/run/opengl-driver"
    ];
  };
}
