import os
import glob

def rename_csv_files(base_folder):
    folders = sorted(os.listdir(base_folder))  # Ana klasör içindeki 37 klasörü sırayla al
    
    prefix_list = ["Aiso", "Biso", "Ciso", "Diso", "Eiso", "Fiso", "Giso", "Hiso", "Iiso", "Jiso", "Kiso", "Liso", "Miso", "Niso", "Oiso", "Piso", "Qiso", "Riso", "Siso", "Tiso", "Uiso", "Viso", "Wiso", "Xiso", "Yiso", "Ziso", "AAiso", "BBiso", "CCiso", "DDiso", "EEiso", "FFiso", "GGiso", "HHiso", "IIiso", "JJiso"]
    
    for i, folder in enumerate(folders):
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue  # Eğer klasör değilse atla
        
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        
        if subfolders:
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder)
                csv_files = sorted(glob.glob(os.path.join(subfolder_path, "*.png")))
                
                prefix = prefix_list[i]  # İlgili prefix'i al
                
                for idx, csv_file in enumerate(csv_files, start=1):
                    new_name = f"{prefix}{idx}.png"  # Yeni isim formatı
                    new_path = os.path.join(subfolder_path, new_name)
                    
                    os.rename(csv_file, new_path)
                    print(f"Renamed: {os.path.basename(csv_file)} -> {new_name}")
        else:
            csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
            
            prefix = prefix_list[i]  # İlgili prefix'i al
            
            for idx, csv_file in enumerate(csv_files, start=1):
                new_name = f"{prefix}{idx}.csv"  # Yeni isim formatı
                new_path = os.path.join(folder_path, new_name)
                
                os.rename(csv_file, new_path)
                print(f"Renamed: {os.path.basename(csv_file)} -> {new_name}")

# Ana klasör yolunu belirleyerek fonksiyonu çağır
target_folder = r"C:\Users\atade\Desktop\erenmodel2\Csv"  # Buraya kendi klasör yolunu yaz
rename_csv_files(target_folder)
