import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import numpy as np

# ==========================
# Load Trained Model
# ==========================

model = load_model('cat_dog_model.h5')

# ==========================
# Function for Prediction
# ==========================

def select_image():

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        messagebox.showwarning(
            "No Image Selected",
            "Please select an image."
        )
        return

    # ==========================
    # Display Image Preview
    # ==========================

    img_preview = Image.open(file_path)

    img_preview.thumbnail((250, 250))

    photo = ImageTk.PhotoImage(img_preview)

    image_label.config(image=photo)

    image_label.image = photo

    # ==========================
    # Model Prediction
    # ==========================

    img = image.load_img(
        file_path,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    score = prediction[0][0]

    # ==========================
    # Display Result
    # ==========================

    if score > 0.5:

        result_title.config(
            text="🐶 DOG DETECTED 🐶",
            fg="#16a34a"
        )

        result_message.config(
            text=
            "⚠️ Tiny AI Brain Still Training\n\n"
            "Sometimes I may confuse fluffy cats with dogs 🐾\n\n"
            "✨ Thanks for testing my AI model!",
            fg="#475569"
        )

    else:

        result_title.config(
            text="🐱 CAT DETECTED 🐱",
            fg="#2563eb"
        )

        result_message.config(
            text=
            "⚠️ Tiny AI Brain Still Training\n\n"
            "Sometimes I may confuse fluffy dogs with cats 🐾\n\n"
            "✨ Thanks for testing my AI model!",
            fg="#475569"
        )

# ==========================
# Main Window
# ==========================

root = tk.Tk()

root.title("AI Pet Classifier")

root.geometry("650x850")

root.configure(bg="#eef4ff")

# ==========================
# Main Card Frame
# ==========================

main_frame = tk.Frame(
    root,
    bg="white"
)

main_frame.place(
    relx=0.5,
    rely=0.5,
    anchor="center",
    width=550,
    height=780
)

# ==========================
# Header
# ==========================

heading = tk.Label(
    main_frame,
    text="🐾 AI Pet Classifier",
    font=("Helvetica", 24, "bold"),
    bg="white",
    fg="#1e293b"
)

heading.pack(pady=20)

subheading = tk.Label(
    main_frame,
    text="Upload an image to identify cats and dogs",
    font=("Helvetica", 11),
    bg="white",
    fg="gray"
)

subheading.pack()

# ==========================
# Image Preview Box
# ==========================

image_frame = tk.Frame(
    main_frame,
    bg="#f8fafc",
    width=300,
    height=300,
    highlightbackground="#dbeafe",
    highlightthickness=2
)

image_frame.pack(pady=25)

image_frame.pack_propagate(False)

image_label = tk.Label(
    image_frame,
    bg="#f8fafc"
)

image_label.pack(expand=True)

# ==========================
# Upload Button
# ==========================

select_btn = tk.Button(
    main_frame,
    text="📂 Upload Image",
    command=select_image,
    width=20,
    height=2,
    bg="#2563eb",
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    cursor="hand2"
)

select_btn.pack(pady=15)

# ==========================
# Result Box
# ==========================

result_frame = tk.Frame(
    main_frame,
    bg="#f8fafc",
    highlightbackground="#dbeafe",
    highlightthickness=1,
    padx=25,
    pady=20
)

result_frame.pack(pady=20)

# Main Result Title

result_title = tk.Label(
    result_frame,
    text="Upload an image to begin",
    font=("Helvetica", 18, "bold"),
    bg="#f8fafc",
    fg="#334155",
    justify="center"
)

result_title.pack(pady=5)

# Result Message

result_message = tk.Label(
    result_frame,
    text="",
    font=("Helvetica", 11),
    bg="#f8fafc",
    fg="#475569",
    justify="center",
    wraplength=420
)

result_message.pack()

# ==========================
# Footer
# ==========================

footer_note = tk.Label(
    main_frame,
    text="Powered by TensorFlow • Keras • Python",
    font=("Helvetica", 10),
    bg="white",
    fg="#94a3b8"
)

footer_note.pack(side="bottom", pady=15)

# ==========================
# Run Application
# ==========================

root.mainloop()