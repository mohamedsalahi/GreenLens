{
  description = "GreenLens: plant disease classification from leaf images (Rust + egui desktop app, TensorFlow VGG16 model)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/aa48d347080940b8a2b8d2f48228674e280a3514";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      guiLibs = with pkgs; [
        libglvnd
        mesa
        wayland
        libxkbcommon
        fontconfig
        freetype
        libX11
        libXcursor
        libXrandr
        libXi
        libXext
      ];

      sharedTools = with pkgs; [
        uv
        python312
        pkg-config
      ];

      # The CUDA pip wheels (nvidia-cublas, nvidia-cudnn, ...) drop their
      # shared libraries under the virtualenv. TensorFlow dlopens them at
      # import time, so put them on the loader path once the venv exists.
      venvCudaHook = ''
        _nv="$PWD/.venv/lib/python3.12/site-packages/nvidia"
        if [ -d "$_nv" ]; then
          for _d in "$_nv"/*/lib; do
            [ -d "$_d" ] && LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
          done
          export LD_LIBRARY_PATH
        fi
        unset _nv _d
      '';
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          name = "greenlens";
          packages = (with pkgs; [
            rustc
            cargo
            clippy
            rustfmt
            stdenv.cc
          ]) ++ sharedTools ++ guiLibs;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath guiLibs + ":/run/opengl-driver/lib";
          shellHook = venvCudaHook;
        };

        # Python/ML shell. The NVIDIA driver runtime (libcuda.so.1) lives in
        # /run/opengl-driver/lib on NixOS; TensorFlow dlopens it at runtime.
        ml = pkgs.mkShell {
          name = "greenlens-ml";
          packages = sharedTools ++ (with pkgs; [ imagemagick ]);
          LD_LIBRARY_PATH = "/run/opengl-driver/lib";
          shellHook = venvCudaHook;
        };
      };
    };
}
