import os

print("== AlexNet MNIST Project ===")
print("1. Train Model")
print("2. Test Model")
choice = input("Choose (1/2):")

if choice == "1":
    os.system("python training/train.py")
elif choice =="2":
    os.system("Python testing/test.py")
else:
    print("Invalid choice!")