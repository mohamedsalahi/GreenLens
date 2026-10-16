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
        };

        # Python/ML shell. The NVIDIA driver runtime (libcuda.so.1) lives in
        # /run/opengl-driver/lib on NixOS; TensorFlow dlopens it at runtime.
        ml = pkgs.mkShell {
          name = "greenlens-ml";
          packages = sharedTools ++ (with pkgs; [ imagemagick ]);
          LD_LIBRARY_PATH = "/run/opengl-driver/lib";
        };
      };
    };
}
