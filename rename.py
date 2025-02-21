import os

def rename_png_files(directory):
    def extract_number(filename):
        return int(''.join(filter(str.isdigit, filename))) if any(c.isdigit() for c in filename) else float('inf')
    
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    png_files.sort(key=extract_number)
    
    for i, filename in enumerate(png_files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"L{i}.png"
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")


# Kullanım
directory = r"C:\Users\atade\Desktop\Eren+Veri\resim\Model_Eğitim_Binary_13-20250221T162508Z-001\Model_Eğitim_Binary_13"  # Mevcut dizini kullanır, gerekirse değiştirin


rename_png_files(directory)


#8-d  9*f 10*g 11h 12K 13L 14M 15N