import cv2
import os
import numpy as np
from face_match import FaceMatcher
from audit_logger import log_event


BASE = os.path.dirname(os.path.abspath(__file__))

ADMIN_PIN = "8087"

ROLE_DATA = {
    "1": ("OPERATOR", os.path.join(BASE, "data", "operator")),
    "2": ("SUPERVISOR", os.path.join(BASE, "data", "supervisor"))
}


def save_user(role, name, embedding, photo):

    base_folder = ROLE_DATA[role][1]

    safe_name = "".join(
        c for c in name
        if c.isalnum() or c in " _-"
    ).strip()

    if not safe_name:
        return False

    folder = os.path.join(
        base_folder,
        safe_name
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return False

    embedding = embedding / norm

    np.save(
        os.path.join(
            folder,
            "embedding.npy"
        ),
        embedding
    )

    cv2.imwrite(
        os.path.join(
            folder,
            "face.jpg"
        ),
        photo
    )

    with open(
        os.path.join(
            folder,
            "profile.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Role: " + ROLE_DATA[role][0] + "\n"
        )

        f.write(
            "Name: " + name
        )

    return True


def register_person(role, name, matcher):

    role_name = ROLE_DATA[role][0]

    log_event(
        "ADMIN",
        role_name + " registration started"
    )

    print()
    print(
        "Starting " +
        role_name +
        " registration..."
    )

    print(
        "Look directly at the camera."
    )

    print(
        "Press Q to cancel."
    )

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        log_event(
            "ADMIN",
            role_name + " registration failed - camera could not be opened"
        )

        print(
            "Camera could not be opened."
        )

        return

    while True:

        ok, frame = cap.read()

        if not ok:
            continue

        faces = matcher.app.get(frame)

        if faces:

            face = max(
                faces,
                key=lambda f:
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                role_name + " FACE DETECTED",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            embedding = matcher.get_embedding(
                frame
            )

            if embedding is not None:

                cv2.putText(
                    frame,
                    "CAPTURING...",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.imshow(
                    "Admin Registration",
                    frame
                )

                cv2.waitKey(800)

                success = save_user(
                    role,
                    name,
                    embedding,
                    frame
                )

                if success:

                    log_event(
                        "ADMIN",
                        role_name +
                        " registered successfully - " +
                        name
                    )

                    print()
                    print("==============================")
                    print(
                        role_name +
                        " REGISTRATION"
                    )
                    print("       SUCCESSFUL")
                    print("==============================")
                    print()
                    print("Name:", name)
                    print("Role:", role_name)
                    print("Biometric data saved.")
                    print()

                else:

                    log_event(
                        "ADMIN",
                        role_name +
                        " registration failed - biometric data could not be saved"
                    )

                    print()
                    print(
                        "Failed to save biometric data."
                    )

                break

        else:

            cv2.putText(
                frame,
                "LOOK AT CAMERA",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            "Admin Registration",
            frame
        )

        key = cv2.waitKey(1) & 255

        if key == ord("q"):

            log_event(
                "ADMIN",
                role_name +
                " registration cancelled - " +
                name
            )

            print()
            print(
                "Registration cancelled."
            )

            break

    cap.release()
    cv2.destroyAllWindows()


def main():

    print()
    print("==============================")
    print("        ADMIN SETUP")
    print("==============================")
    print()

    pin = input(
        "Enter Admin PIN: "
    ).strip()

    if pin != ADMIN_PIN:

        log_event(
            "ADMIN",
            "Access denied - invalid PIN"
        )

        print()
        print("ACCESS DENIED")
        print()

        return

    log_event(
        "ADMIN",
        "Admin authentication successful"
    )

    print()
    print(
        "Admin authentication successful."
    )

    print()
    print("1. REGISTER OPERATOR")
    print("2. REGISTER SUPERVISOR")
    print()

    choice = input(
        "Enter choice (1/2): "
    ).strip()

    if choice not in ROLE_DATA:

        log_event(
            "ADMIN",
            "Invalid registration role selected"
        )

        print()
        print("Invalid choice.")
        return

    role_name = ROLE_DATA[choice][0]

    log_event(
        "ADMIN",
        role_name +
        " registration selected"
    )

    print()
    print(
        "Selected:",
        role_name
    )

    name = input(
        "Enter " +
        role_name.title() +
        " Name: "
    ).strip()

    if not name:

        log_event(
            "ADMIN",
            role_name +
            " registration failed - invalid name"
        )

        print()
        print("Invalid name.")
        return

    existing = os.path.join(
        ROLE_DATA[choice][1],
        name,
        "embedding.npy"
    )

    if os.path.exists(existing):

        log_event(
            "ADMIN",
            role_name +
            " registration rejected - already registered - " +
            name
        )

        print()
        print(
            role_name +
            " is already registered."
        )

        return

    print()
    print(
        "Loading face recognition model..."
    )

    try:

        matcher = FaceMatcher()

    except Exception as e:

        log_event(
            "ADMIN",
            role_name +
            " registration failed - face recognition error"
        )

        print()
        print(
            "Face recognition error:"
        )
        print(e)

        return

    register_person(
        choice,
        name,
        matcher
    )


if __name__ == "__main__":
    main()