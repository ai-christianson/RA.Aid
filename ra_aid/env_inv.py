import os
import platform
import shutil
import subprocess
from pathlib import Path

class EnvDiscovery:
    def __init__(self):
        # Structured results dictionary.
        self.results = {
            "os": {},
            "cli_tools": {},
            "python": {"installations": [], "env_tools": {}},
            "package_managers": {},
            "libraries": {},
            "node": {}
        }
        # Common CLI tools. Added additional critical dev tools.
        self._cli_tool_names = [
            "fd", "rg", "fzf", "git", "g++", "gcc", "clang", "cmake", "make",
            "pkg-config", "ninja", "autoconf", "automake", "libtool", "meson", "scons"
        ]
        # Python environment tools.
        self._py_env_tools = {
            "virtualenv": "virtualenv",
            "uv": "uv",
            "pipenv": "pipenv",
            "poetry": "poetry",
            "conda": "conda",
            "pyenv": "pyenv",
            "pipx": "pipx"
        }
        # Package managers.
        self._package_managers = [
            "apt", "apt-get", "dnf", "yum", "pacman", "paru", "zypper",
            "brew", "winget", "choco"
        ]
        # Expanded libraries detection list.
        # Each entry maps a library key to a dict with possible keys:
        #   - 'pkg': pkg-config name if available.
        #   - 'headers': list of header paths relative to common include directories.
        self._libraries = {
            # Graphics & Game Dev:
            "SDL2": {"pkg": "sdl2", "headers": ["SDL2/SDL.h", "SDL.h"]},
            "OpenGL": {"pkg": "gl", "headers": ["GL/gl.h", "OpenGL/gl.h"]},
            "Vulkan": {"pkg": "vulkan", "headers": ["vulkan/vulkan.h"]},
            "DirectX": {"headers": []},  # Windows only; detection via headers is non-trivial.
            "GLFW": {"pkg": "glfw3", "headers": ["GLFW/glfw3.h"]},
            "Raylib": {"pkg": "raylib", "headers": ["raylib.h"]},
            "SFML": {"headers": ["SFML/Graphics.hpp", "SFML/Window.hpp"]},
            "Allegro": {"pkg": "allegro", "headers": ["allegro5/allegro.h"]},
            "OGRE": {"headers": ["OGRE/Ogre.h"]},
            "Irrlicht": {"headers": ["irrlicht.h"]},
            "bgfx": {"headers": ["bgfx/bgfx.h"]},
            "Magnum": {"headers": ["Magnum/Platform/GlfwApplication.h"]},
            "Assimp": {"pkg": "assimp", "headers": ["assimp/Importer.hpp"]},
            "DearImGui": {"headers": ["imgui.h"]},
            "Cairo": {"pkg": "cairo", "headers": ["cairo.h"]},
            "NanoVG": {"headers": ["nanovg.h"]},
            # Physics Engines:
            "Bullet": {"headers": ["bullet/btBulletDynamicsCommon.h"]},
            "PhysX": {"headers": []},
            "ODE": {"pkg": "ode", "headers": ["ode/ode.h"]},
            "Box2D": {"pkg": "box2d", "headers": ["box2d/box2d.h"]},
            "JoltPhysics": {"headers": ["Jolt/Jolt.h"]},
            "MuJoCo": {"headers": ["mujoco.h"]},
            "Newton": {"pkg": "newton", "headers": ["Newton/Newton.h"]},
            # Math & Linear Algebra:
            "Eigen": {"headers": ["Eigen/Dense"]},
            "GLM": {"headers": ["glm/glm.hpp"]},
            "Armadillo": {"pkg": "armadillo", "headers": ["armadillo"]},
            "BLAS": {"headers": []},
            "LAPACK": {"headers": []},
            "OpenBLAS": {"headers": []},
            "IntelMKL": {"headers": []},
            "Boost_uBLAS": {"headers": ["boost/numeric/ublas/matrix.hpp"]},
            "Blaze": {"headers": ["blaze/Blaze.h"]},
            "Blitz++": {"headers": ["blitz/array.h"]},
            "xtensor": {"headers": ["xtensor/xarray.hpp"]},
            "GSL": {"pkg": "gsl", "headers": ["gsl/gsl_errno.h"]},
            # Machine Learning & AI:
            "TensorFlow": {"pkg": "tensorflow", "headers": ["tensorflow/c/c_api.h"]},
            "PyTorch": {"pkg": "torch", "headers": []},
            "ONNX": {"pkg": "onnx", "headers": []},
            "OpenCV": {"pkg": "opencv", "headers": ["opencv2/opencv.hpp"]},
            "scikit-learn": {"headers": []},
            "Caffe": {"headers": ["caffe/caffe.hpp"]},
            "MXNet": {"headers": ["mxnet-cpp/MxNetCpp.h"]},
            "XGBoost": {"pkg": "xgboost", "headers": []},
            "LightGBM": {"headers": []},
            "dlib": {"pkg": "dlib", "headers": ["dlib/dlib.h"]},
            "OpenVINO": {"headers": []},
            "TensorRT": {"headers": []},
            # Networking & Communication:
            "Boost_Asio": {"headers": ["boost/asio.hpp"]},
            "libcurl": {"pkg": "libcurl", "headers": ["curl/curl.h"]},
            "ZeroMQ": {"pkg": "libzmq", "headers": ["zmq.h"]},
            "gRPC": {"pkg": "grpc", "headers": ["grpc/grpc.h"]},
            "Thrift": {"headers": ["thrift/Thrift.h"]},
            "libevent": {"pkg": "libevent", "headers": ["event2/event.h"]},
            "libuv": {"pkg": "libuv", "headers": ["uv.h"]},
            "Boost_Beast": {"headers": ["boost/beast.hpp"]},
            "libwebsockets": {"pkg": "libwebsockets", "headers": ["libwebsockets.h"]},
            "MQTT": {"pkg": "paho-mqtt3c", "headers": ["MQTTClient.h"]},
            "APR": {"pkg": "apr-1", "headers": ["apr.h"]},
            "nng": {"pkg": "nng", "headers": ["nng/nng.h"]},
            # Compression & Encoding:
            "zlib": {"pkg": "zlib", "headers": ["zlib.h"]},
            "LZ4": {"pkg": "lz4", "headers": ["lz4.h"]},
            "Zstd": {"pkg": "zstd", "headers": ["zstd.h"]},
            "Brotli": {"pkg": "brotli", "headers": ["brotli/decode.h"]},
            "bzip2": {"pkg": "bzip2", "headers": ["bzlib.h"]},
            "xz": {"pkg": "liblzma", "headers": ["lzma.h"]},
            "Snappy": {"pkg": "snappy", "headers": ["snappy.h"]},
            "libpng": {"pkg": "libpng", "headers": ["png.h"]},
            "libjpeg": {"pkg": "libjpeg", "headers": ["jpeglib.h"]},
            "libtiff": {"pkg": "libtiff-4", "headers": ["tiffio.h"]},
            "libwebp": {"pkg": "libwebp", "headers": ["webp/encode.h"]},
            "FFmpeg": {"pkg": "libavcodec", "headers": ["libavcodec/avcodec.h"]},
            "GStreamer": {"pkg": "gstreamer-1.0", "headers": ["gst/gst.h"]},
            "libogg": {"pkg": "libogg", "headers": ["ogg/ogg.h"]},
            "libvorbis": {"pkg": "vorbis", "headers": ["vorbis/codec.h"]},
            "libFLAC": {"pkg": "flac", "headers": ["FLAC/stream_encoder.h"]},
            # Databases & Data Storage:
            "SQLite": {"pkg": "sqlite3", "headers": ["sqlite3.h"]},
            "PostgreSQL": {"pkg": "libpq", "headers": ["libpq-fe.h"]},
            "MySQL": {"pkg": "mysqlclient", "headers": ["mysql.h"]},
            "Redis": {"headers": []},
            "LevelDB": {"headers": ["leveldb/db.h"]},
            "RocksDB": {"headers": ["rocksdb/db.h"]},
            "BerkeleyDB": {"headers": ["db.h"]},
            "HDF5": {"pkg": "hdf5", "headers": ["hdf5.h"]},
            # Parallel Computing & GPU:
            "OpenMP": {"headers": []},
            "MPI": {"pkg": "mpi", "headers": ["mpi.h"]},
            "CUDA": {"pkg": "cuda", "headers": ["cuda.h"]},
            "OpenCL": {"pkg": "OpenCL", "headers": ["CL/cl.h"]},
            "oneAPI": {"headers": []},
            "HIP": {"headers": []},
            "OpenACC": {"headers": []},
            "TBB": {"pkg": "tbb", "headers": ["tbb/tbb.h"]},
            "cuDNN": {"headers": []},
            "MicrosoftMPI": {"headers": []},
            # Cryptography & Security:
            "OpenSSL": {"pkg": "openssl", "headers": ["openssl/ssl.h"]},
            "LibreSSL": {"pkg": "openssl", "headers": ["openssl/ssl.h"]},
            "BoringSSL": {"headers": []},
            "libsodium": {"pkg": "sodium", "headers": ["sodium.h"]},
            "Crypto++": {"headers": ["cryptopp/cryptlib.h"]},
            "Botan": {"headers": ["botan/botan.h"]},
            "GnuTLS": {"pkg": "gnutls", "headers": ["gnutls/gnutls.h"]},
            "mbedTLS": {"pkg": "mbedtls", "headers": ["mbedtls/ssl.h"]},
            "wolfSSL": {"pkg": "wolfssl", "headers": ["wolfssl/options.h"]},
            # Scripting & Embedding:
            "Python_C_API": {"headers": ["Python.h"]},
            "Lua": {"pkg": "lua", "headers": ["lua.h"]},
            "LuaJIT": {"pkg": "luajit", "headers": ["luajit.h"]},
            "V8": {"headers": ["v8.h"]},
            "Duktape": {"headers": ["duktape.h"]},
            "SpiderMonkey": {"headers": ["jsapi.h"]},
            "JavaScriptCore": {"headers": ["JavaScriptCore/JavaScript.h"]},
            "ChakraCore": {"headers": ["ChakraCore.h"]},
            "Tcl": {"pkg": "tcl", "headers": ["tcl.h"]},
            "Guile": {"headers": ["libguile.h"]},
            "Mono": {"headers": ["mono/jit/jit.h"]},
            # Audio & Multimedia:
            "OpenAL": {"pkg": "openal", "headers": ["AL/al.h"]},
            "PortAudio": {"pkg": "portaudio-2.0", "headers": ["portaudio.h"]},
            "FMOD": {"headers": []},
            "SoLoud": {"headers": ["soloud.h"]},
            "RtAudio": {"headers": ["RtAudio.h"]},
            "SDL_mixer": {"pkg": "SDL2_mixer", "headers": ["SDL2/SDL_mixer.h"]},
            "OpenAL_Soft": {"pkg": "openal", "headers": ["AL/al.h"]},
            "libsndfile": {"pkg": "sndfile", "headers": ["sndfile.h"]},
            "Jack": {"pkg": "jack", "headers": ["jack/jack.h"]},
            # Dev Utilities & Frameworks:
            "Boost": {"headers": ["boost/config.hpp"]},
            "Qt": {"headers": ["QtCore/QtCore"]},
            "wxWidgets": {"headers": ["wx/wx.h"]},
            "GTK": {"pkg": "gtk+-3.0", "headers": ["gtk/gtk.h"]},
            "ncurses": {"pkg": "ncurses", "headers": ["ncurses.h"]},
            "Poco": {"headers": ["Poco/Foundation.h"]},
            "ICU": {"pkg": "icu-uc", "headers": ["unicode/utypes.h"]},
            "RapidJSON": {"headers": ["rapidjson/document.h"]},
            "nlohmann_json": {"headers": ["nlohmann/json.hpp"]},
            "json-c": {"pkg": "json-c", "headers": ["json-c/json.h"]},
            "YAML_cpp": {"headers": ["yaml-cpp/yaml.h"]},
            "spdlog": {"headers": ["spdlog/spdlog.h"]},
            "log4cxx": {"headers": ["log4cxx/logger.h"]},
            "glog": {"headers": ["glog/logging.h"]},
            "GoogleTest": {"headers": ["gtest/gtest.h"]},
            "BoostTest": {"headers": ["boost/test/unit_test.hpp"]},
            "pkg-config": {"headers": []},
            "CMake": {"headers": []},
            "GLib": {"pkg": "glib-2.0", "headers": ["glib.h"]}
        }
        # List of common include directories to search for headers.
        # Expanded to cover multiple common Homebrew paths on macOS and Linuxbrew.
        self._include_paths = [
            Path("/usr/include"),
            Path("/usr/local/include"),
            Path("/opt/homebrew/include"),
            Path("/home/linuxbrew/.linuxbrew/include"),
            Path("/usr/local/Homebrew/include")
        ]
        # Linux distribution info.
        self._distro = {}
        if platform.system() == "Linux":
            self._distro = self._get_linux_distro()

    def _get_linux_distro(self):
        distro = {}
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if "=" not in line:
                        continue
                    key, val = line.strip().split("=", 1)
                    distro[key] = val.strip('"')
        except FileNotFoundError:
            pass
        return distro

    def discover(self):
        self._detect_os()
        self._detect_cli_tools()
        self._detect_python()
        self._detect_python_env_tools()
        self._detect_package_managers()
        self._detect_libraries()
        self._detect_node()
        return self.results

    def _detect_os(self):
        os_type = platform.system()
        os_info = {}
        if os_type == "Windows":
            os_info["name"] = "Windows"
            os_info["wsl"] = False
        elif os_type == "Linux":
            release = platform.uname().release
            if "Microsoft" in release or release.lower().endswith("microsoft"):
                os_info["name"] = "Linux (WSL)"
                os_info["wsl"] = True
            else:
                os_info["name"] = "Linux"
                os_info["wsl"] = False
            if self._distro:
                name = self._distro.get("PRETTY_NAME") or self._distro.get("NAME")
                version = self._distro.get("VERSION_ID") or self._distro.get("VERSION")
                if name:
                    os_info["distro"] = name
                if version:
                    os_info["distro_version"] = version
        elif os_type == "Darwin":
            os_info["name"] = "macOS"
            os_info["wsl"] = False
        else:
            os_info["name"] = os_type
            os_info["wsl"] = False
        self.results["os"] = os_info

    def _detect_cli_tools(self):
        tools_found = {}
        for tool in self._cli_tool_names:
            path = shutil.which(tool)
            if path:
                version = None
                if tool in ("g++", "gcc", "clang", "git"):
                    try:
                        out = subprocess.check_output([tool, "--version"], text=True, stderr=subprocess.STDOUT, timeout=1)
                        version = out.splitlines()[0].strip()
                    except Exception:
                        version = None
                tools_found[tool] = {"found": True}
                if version:
                    tools_found[tool]["version"] = version
            else:
                tools_found[tool] = {"found": False}
        self.results["cli_tools"] = tools_found

    def _detect_python(self):
        installations = []
        if platform.system() == "Windows":
            launcher = shutil.which("py")
            if launcher:
                try:
                    out = subprocess.check_output([launcher, "-0p"], text=True, timeout=2)
                    for line in out.splitlines():
                        line = line.strip()
                        if not line or not line.startswith("-V:"):
                            continue
                        after = line.split(":", 1)[1]
                        parts = after.strip().split(None, 1)
                        ver_str = parts[0].lstrip("v")
                        py_path = parts[1] if len(parts) > 1 else ""
                        installations.append({"version": ver_str, "path": py_path})
                except subprocess.CalledProcessError:
                    pass
            if not installations:
                try:
                    out = subprocess.check_output(["where", "python"], text=True, timeout=2)
                    for path in out.splitlines():
                        path = path.strip()
                        if path and Path(path).name.lower().startswith("python"):
                            ver = self._get_python_version(path)
                            installations.append({"version": ver, "path": path})
                except Exception:
                    pass
        else:
            common_names = ["python3", "python", "python2"]
            for major in [2, 3]:
                for minor in range(0, 15):
                    common_names.append(f"python{major}.{minor}")
            seen_paths = set()
            for name in common_names:
                path = shutil.which(name)
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    ver = self._get_python_version(path)
                    installations.append({"version": ver, "path": path})

        installations = sorted(installations, key=lambda x: x.get("version", "") or "")
        self.results["python"]["installations"] = installations

    def _get_python_version(self, python_path):
        try:
            out = subprocess.check_output([python_path, "--version"], stderr=subprocess.STDOUT, text=True, timeout=1)
            ver = out.strip().split()[1]
            return ver
        except Exception:
            return None

    def _detect_python_env_tools(self):
        env_tools_status = {}
        venv_available = any(inst for inst in self.results["python"]["installations"]
                             if inst.get("version") and inst["version"][0] == '3')
        env_tools_status["venv"] = {"available": venv_available, "built_in": True}
        for tool, display_name in self._py_env_tools.items():
            found_path = shutil.which(tool)
            if found_path:
                version = None
                try:
                    if tool == "pyenv":
                        out = subprocess.check_output([tool, "--version"], text=True, timeout=1)
                        version = out.strip().split()[-1]
                    elif tool in ("pipenv", "poetry", "conda", "pipx", "uv"):
                        out = subprocess.check_output([tool, "--version"], text=True, timeout=2)
                        version = out.strip().split()[-1]
                    elif tool == "virtualenv":
                        out = subprocess.check_output([tool, "--version"], text=True, timeout=2)
                        version = out.strip()
                except Exception:
                    version = None
                env_tools_status[display_name] = {"installed": True}
                if version:
                    env_tools_status[display_name]["version"] = version
            else:
                env_tools_status[display_name] = {"installed": False}
        self.results["python"]["env_tools"] = env_tools_status

    def _detect_package_managers(self):
        pkg_status = {}
        for mgr in self._package_managers:
            if platform.system() == "Windows":
                if mgr in ("apt", "apt-get", "dnf", "yum", "pacman", "paru", "zypper", "brew"):
                    continue
            if platform.system() == "Darwin":
                if mgr in ("apt", "apt-get", "dnf", "yum", "pacman", "paru", "zypper", "winget", "choco"):
                    continue
            if platform.system() == "Linux" and self._distro:
                distro_id = self._distro.get("ID", "").lower()
                if distro_id:
                    if distro_id in ("debian", "ubuntu", "linuxmint"):
                        if mgr in ("pacman", "paru", "yum", "dnf", "zypper"):
                            continue
                    if distro_id in ("fedora", "centos", "rhel", "rocky", "alma"):
                        if mgr in ("apt", "apt-get", "pacman", "paru", "zypper"):
                            continue
                    if distro_id in ("arch", "manjaro", "endeavouros"):
                        if mgr in ("apt", "apt-get", "dnf", "yum", "zypper"):
                            continue
                    if distro_id in ("opensuse", "suse"):
                        if mgr in ("apt", "apt-get", "dnf", "yum", "pacman", "paru"):
                            continue
            path = shutil.which(mgr)
            pkg_status[mgr] = {"found": bool(path)}
            if path:
                version = None
                try:
                    if mgr in ("brew", "winget", "choco"):
                        out = subprocess.check_output([mgr, "--version"], text=True, timeout=3)
                        version_line = out.splitlines()[0].strip()
                        version = version_line
                    elif mgr in ("apt", "apt-get", "pacman", "paru", "dnf", "yum", "zypper"):
                        out = subprocess.check_output([mgr, "--version"], text=True, timeout=2)
                        version_line = out.splitlines()[0].strip()
                        version = version_line
                except Exception:
                    version = None
                if version:
                    pkg_status[mgr]["version"] = version
        self.results["package_managers"] = pkg_status

    def _detect_libraries(self):
        libs_found = {}
        have_pkg_config = bool(shutil.which("pkg-config"))
        for lib, info in self._libraries.items():
            lib_info = {"found": False}
            found = False
            ver = None
            cflags = None
            libs_flags = None
            header_paths = []
            if have_pkg_config and info.get("pkg"):
                pkg_name = info["pkg"]
                try:
                    subprocess.check_output(["pkg-config", "--exists", pkg_name],
                                            stderr=subprocess.DEVNULL, timeout=1)
                    found = True
                    try:
                        ver = subprocess.check_output(
                            ["pkg-config", "--modversion", pkg_name],
                            text=True, timeout=1
                        ).strip()
                    except Exception:
                        ver = None
                    try:
                        cflags = subprocess.check_output(
                            ["pkg-config", "--cflags", pkg_name],
                            text=True, timeout=1
                        ).strip()
                    except Exception:
                        cflags = None
                    try:
                        libs_flags = subprocess.check_output(
                            ["pkg-config", "--libs", pkg_name],
                            text=True, timeout=1
                        ).strip()
                    except Exception:
                        libs_flags = None
                except subprocess.CalledProcessError:
                    found = False
            if not found and info.get("headers"):
                for header in info["headers"]:
                    for inc_dir in self._include_paths:
                        header_file = inc_dir / header
                        if header_file.exists():
                            found = True
                            header_paths.append(str(header_file))
            lib_info["found"] = found
            if ver:
                lib_info["version"] = ver
            if cflags:
                lib_info["cflags"] = cflags
            if libs_flags:
                lib_info["libs"] = libs_flags
            if header_paths:
                lib_info["header_paths"] = header_paths
            libs_found[lib] = lib_info
        self.results["libraries"] = libs_found

    def _detect_node(self):
        node_info = {}
        node_path = shutil.which("node")
        if node_path:
            try:
                out = subprocess.check_output(["node", "--version"], text=True, timeout=1)
                node_info["node_version"] = out.strip()
            except Exception:
                node_info["node_version"] = "found"
        else:
            node_info["node_version"] = None
        npm_path = shutil.which("npm")
        if npm_path:
            try:
                out = subprocess.check_output(["npm", "--version"], text=True, timeout=1)
                node_info["npm_version"] = out.strip()
            except Exception:
                node_info["npm_version"] = "found"
        else:
            node_info["npm_version"] = None
        nvm_installed = False
        nvm_version = None
        if platform.system() == "Windows":
            if shutil.which("nvm"):
                nvm_installed = True
                try:
                    out = subprocess.check_output(["nvm", "version"], text=True, timeout=2)
                    nvm_version = out.strip()
                except Exception:
                    nvm_version = None
        else:
            if os.environ.get("NVM_DIR") or Path.home().joinpath(".nvm").exists():
                nvm_installed = True
        node_info["nvm_installed"] = nvm_installed
        if nvm_version:
            node_info["nvm_version"] = nvm_version
        self.results["node"] = node_info

    def format_markdown(self):
        os_info = self.results.get("os", {})
        lines = []
        # OS Section
        os_section = f"**Operating System:** {os_info.get('name')}"
        if os_info.get("distro"):
            os_section += f" ({os_info['distro']}"
            if os_info.get("distro_version"):
                os_section += f" {os_info['distro_version']}"
            os_section += ")"
        lines.append(os_section)
        if os_info.get("wsl"):
            lines.append("- Running under WSL")
        lines.append("")
        # CLI Tools Section - output as one list.
        cli_found = []
        for tool, status in self.results.get("cli_tools", {}).items():
            if status.get("found"):
                if status.get("version"):
                    cli_found.append(f"{tool} ({status['version']})")
                else:
                    cli_found.append(tool)
        if cli_found:
            lines.append("**Found CLI developer tools:** " + ", ".join(cli_found))
        else:
            lines.append("**Found CLI developer tools:** None")
        lines.append("")
        # Python Section
        py_installs = self.results.get("python", {}).get("installations", [])
        env_tools = self.results.get("python", {}).get("env_tools", {})
        lines.append("**Python Environments:**")
        if py_installs:
            for py in py_installs:
                ver = py.get("version") or "unknown version"
                path = py.get("path")
                lines.append(f"- Python {ver} at `{path}`")
        else:
            lines.append("- No Python interpreter found")
        for tool, info in env_tools.items():
            if tool == "venv":
                available = info.get("available", False)
                lines.append(f"- venv (builtin): {'available' if available else 'not available'}")
            else:
                installed = info.get("installed", False)
                ver = info.get("version")
                if installed:
                    if ver:
                        lines.append(f"- {tool}: installed (version {ver})")
                    else:
                        lines.append(f"- {tool}: installed")
                else:
                    lines.append(f"- {tool}: not installed")
        lines.append("")
        # Package Managers Section
        pkg_mgrs = self.results.get("package_managers", {})
        lines.append("**Package Managers:**")
        any_pkg = False
        for mgr, info in pkg_mgrs.items():
            if not info.get("found"):
                continue
            any_pkg = True
            ver = info.get("version")
            if ver:
                lines.append(f"- {mgr}: found ({ver})")
            else:
                lines.append(f"- {mgr}: found")
        if not any_pkg:
            lines.append("- *(No common package managers found)*")
        lines.append("")
        # Libraries Section
        libs = self.results.get("libraries", {})
        lines.append("**Developer Libraries:**")
        found_libs = []
        not_found_libs = []
        for lib, info in libs.items():
            if info.get("found"):
                line = f"- {lib}: installed"
                if info.get("version"):
                    line += f" (version {info['version']})"
                if info.get("cflags"):
                    line += f", cflags: `{info['cflags']}`"
                if info.get("libs"):
                    line += f", libs: `{info['libs']}`"
                if info.get("header_paths"):
                    line += f", headers: {', '.join(info['header_paths'])}"
                found_libs.append(line)
            else:
                not_found_libs.append(lib)
        lines.extend(found_libs)
        if not_found_libs:
            lines.append(f"- Not found: {', '.join(sorted(not_found_libs))}")
        lines.append("")
        # Node.js Section
        node = self.results.get("node", {})
        lines.append("**Node.js and Related:**")
        node_ver = node.get("node_version")
        npm_ver = node.get("npm_version")
        nvm_inst = node.get("nvm_installed")
        nvm_ver = node.get("nvm_version")
        if node_ver:
            lines.append(f"- Node.js: {node_ver}")
        else:
            lines.append("- Node.js: not installed")
        if npm_ver:
            lines.append(f"- npm: version {npm_ver}")
        else:
            lines.append("- npm: not installed")
        if nvm_inst:
            if nvm_ver:
                lines.append(f"- nvm: installed (version {nvm_ver})")
            else:
                lines.append("- nvm: installed")
        else:
            lines.append("- nvm: not installed")
        lines.append("")
        return "\n".join(lines)

if __name__ == "__main__":
    env = EnvDiscovery()
    env.discover()
    print(env.format_markdown())
