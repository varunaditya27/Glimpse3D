import sys

def read_log():
    try:
        # Try UTF-16 LE first (Powershell default)
        with open("backend.log", "r", encoding="utf-16-le") as f:
            print(f.read())
    except UnicodeError:
        # Fallback to utf-8 if empty or different
        try:
            with open("backend.log", "r", encoding="utf-8") as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading log: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_log()
