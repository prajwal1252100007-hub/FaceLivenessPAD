import os
import numpy as np
import cv2

from main import App
from audit_logger import log_event


BASE = os.path.dirname(os.path.abspath(__file__))
SUPERVISOR_DATA = os.path.join(BASE, "data", "supervisor")


class SupervisorApp(App):

    def folder(self):
        safe = "".join(
            c for c in self.name
            if c.isalnum() or c in " _-"
        ).strip()

        return os.path.join(
            SUPERVISOR_DATA,
            safe
        )

    def load_user(self):
        path = os.path.join(
            self.folder(),
            "embedding.npy"
        )

        if not os.path.exists(path):
            return None

        x = np.load(path).astype(np.float32)

        norm = np.linalg.norm(x)

        if norm == 0:
            return None

        return x / norm


def main():

    print()
    print("==============================")
    print("      SUPERVISOR LOGIN")
    print("==============================")
    print()

    name = input(
        "Enter Supervisor Name: "
    ).strip()

    if not name:

        log_event(
            "SUPERVISOR",
            "Authentication failed - invalid name"
        )

        print()
        print("Invalid name.")
        return

    log_event(
        "SUPERVISOR",
        "Login started"
    )

    safe = "".join(
        c for c in name
        if c.isalnum() or c in " _-"
    ).strip()

    folder = os.path.join(
        SUPERVISOR_DATA,
        safe
    )

    embedding_path = os.path.join(
        folder,
        "embedding.npy"
    )

    if not os.path.exists(
        embedding_path
    ):

        log_event(
            "SUPERVISOR",
            "Authentication failed - supervisor not registered"
        )

        print()
        print("Supervisor not registered.")
        print()

        return

    print()
    print(
        "Starting Supervisor Authentication..."
    )

    log_event(
        "SUPERVISOR",
        "Authentication started"
    )

    app = None

    try:

        app = SupervisorApp(
            "REGISTERED",
            name
        )

        cv2.namedWindow(
            "Face Liveness + PAD",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Face Liveness + PAD",
            900,
            405
        )

        app.run()

        log_event(
            "SUPERVISOR",
            "Authentication session completed"
        )

    except Exception as e:

        log_event(
            "SUPERVISOR",
            f"Authentication error - {e}"
        )

        print()
        print(
            "SUPERVISOR LOGIN ERROR:"
        )
        print(e)

    finally:

        if app:

            try:
                app.running = False
                app.cam.stop()
                app.mesh.close()
            except:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()