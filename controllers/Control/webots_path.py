import os
import sys
from pathlib import Path


def configure_webots_python_path() -> Path:
    webots_home = os.getenv("WEBOTS_HOME")
    if not webots_home:
        default_home = Path(r"C:\Program Files\Webots")
        if default_home.exists():
            webots_home = str(default_home)
            os.environ["WEBOTS_HOME"] = webots_home
        else:
            raise RuntimeError(
                "WEBOTS_HOME is not set. Set it to your Webots install directory, "
                "for example C:\\Program Files\\Webots."
            )

    controller_python = Path(webots_home) / "lib" / "controller" / "python"
    if not controller_python.exists():
        raise RuntimeError(f"Webots Python controller path not found: {controller_python}")

    controller_path = str(controller_python)
    if controller_path not in sys.path:
        sys.path.insert(0, controller_path)

    pythonpath = os.environ.get("PYTHONPATH")
    paths = pythonpath.split(os.pathsep) if pythonpath else []
    if controller_path not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([controller_path, *paths])

    return controller_python
