import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def run_file(filename):
    path = os.path.join(BASE, filename)

    subprocess.run(
        [sys.executable, path],
        cwd=BASE
    )

while True:

    print()
    print("========================================")
    print("       FACE LIVENESS + PAD SYSTEM")
    print("========================================")
    print()
    print("1. NEW RESIDENT")
    print("2. REGISTERED RESIDENT")
    print("3. OPERATOR LOGIN")
    print("4. SUPERVISOR LOGIN")
    print("5. ADMIN SETUP")
    print("6. EXIT")
    print()

    choice = input("Enter choice: ").strip()

    if choice == "1":
        run_file("main.py")

    elif choice == "2":
        run_file("main.py")

    elif choice == "3":
        run_file("operator_login.py")

    elif choice == "4":
        run_file("supervisor_login.py")

    elif choice == "5":
        run_file("admin_setup.py")

    elif choice == "6":
        print("Exiting...")
        break

    else:
        print("Invalid choice.")