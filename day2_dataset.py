from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def get_image_paths(directory: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    )


def load_image(path: Path):
    try:
        image = cv2.imread(str(path))
        return image
    except Exception:
        return None


def summarize_class(class_dir: Path, max_samples=5):
    image_paths = get_image_paths(class_dir)
    total_images = len(image_paths)
    corrupt_files = []
    valid_dims = []
    samples = []

    for path in image_paths:
        image = load_image(path)
        if image is None:
            corrupt_files.append(path)
            continue

        height, width = image.shape[:2]
        valid_dims.append((width, height))
        if len(samples) < max_samples:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            samples.append((path.name, rgb_image))

    return total_images, corrupt_files, valid_dims, samples


def format_average_dimensions(valid_dims):
    if not valid_dims:
        return "N/A", "N/A"
    avg_width = sum(width for width, _ in valid_dims) / len(valid_dims)
    avg_height = sum(height for _, height in valid_dims) / len(valid_dims)
    return f"{avg_width:.1f}", f"{avg_height:.1f}"


def write_summary(report_lines, output_path: Path):
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved summary report to: {output_path}")


def show_samples(sample_data, class_names):
    if not any(samples for _, samples in sample_data.items()):
        print("No sample images were found to display.")
        return

    rows = len(class_names)
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    if rows == 1:
        axes = [axes]

    for row_idx, class_name in enumerate(class_names):
        samples = sample_data[class_name]
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
            ax.axis("off")
            if col_idx < len(samples):
                filename, image = samples[col_idx]
                ax.imshow(image)
                ax.set_title(f"{class_name}\n{filename}", fontsize=9)
            else:
                ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    class_names = ["with_mask", "without_mask"]
    output_file = project_root / "dataset_summary.txt"

    report_lines = ["Dataset Summary Report", "======================", ""]
    sample_data = {}

    for class_name in class_names:
        class_dir = data_root / class_name
        report_lines.append(f"Class: {class_name}")

        if not class_dir.exists() or not class_dir.is_dir():
            report_lines.append("  Directory not found.")
            report_lines.append("")
            sample_data[class_name] = []
            continue

        total_images, corrupt_files, valid_dims, samples = summarize_class(class_dir)
        avg_width, avg_height = format_average_dimensions(valid_dims)

        report_lines.append(f"  Total images: {total_images}")
        report_lines.append(f"  Average width: {avg_width}")
        report_lines.append(f"  Average height: {avg_height}")
        report_lines.append(f"  Corrupt/unreadable files: {len(corrupt_files)}")
        for corrupt_path in corrupt_files:
            report_lines.append(f"    - {corrupt_path}")
        report_lines.append("")

        sample_data[class_name] = samples

    write_summary(report_lines, output_file)
    print("\n".join(report_lines))
    show_samples(sample_data, class_names)


if __name__ == "__main__":
    main()
