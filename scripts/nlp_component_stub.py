def generate_safety_message(detected_items):
    if not detected_items:
        return "No PPE items detected."

    items = ", ".join(detected_items)
    return f"Detected PPE items: {items}."


if __name__ == "__main__":
    sample = ["hardhat", "vest"]
    print(generate_safety_message(sample))