"""
Script untuk membandingkan dua file submission CSV.
Menghitung berapa banyak row yang berbeda dan persentase perbedaannya.

Usage:
    python compare_submissions.py submission1.csv submission2.csv
    atau
    python compare_submissions.py
    (akan diminta input path file)
"""

import pandas as pd
import sys
import os


def compare_submissions(file1_path, file2_path):
    """
    Membandingkan dua file submission CSV.
    
    Args:
        file1_path: Path ke file submission pertama
        file2_path: Path ke file submission kedua
    
    Returns:
        Dictionary dengan hasil perbandingan
    """
    try:
        # Baca kedua file
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        print("="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        # Validasi struktur file
        print(f"\nFile 1: {os.path.basename(file1_path)}")
        print(f"  Shape: {df1.shape}")
        print(f"  Columns: {list(df1.columns)}")
        
        print(f"\nFile 2: {os.path.basename(file2_path)}")
        print(f"  Shape: {df2.shape}")
        print(f"  Columns: {list(df2.columns)}")
        
        # Cek apakah jumlah rows sama
        if len(df1) != len(df2):
            print(f"\n⚠️ WARNING: Jumlah rows berbeda!")
            print(f"  File 1: {len(df1)} rows")
            print(f"  File 2: {len(df2)} rows")
            print(f"  Akan membandingkan hanya rows yang ada di kedua file.")
        
        # Cek apakah kolom sama
        if list(df1.columns) != list(df2.columns):
            print(f"\n⚠️ WARNING: Kolom tidak sama!")
            common_cols = set(df1.columns) & set(df2.columns)
            print(f"  Kolom yang sama: {list(common_cols)}")
        
        # Asumsikan kolom pertama adalah ID dan kolom kedua adalah prediction
        id_col = df1.columns[0]
        pred_col = df1.columns[1]
        
        # Merge berdasarkan ID untuk perbandingan yang akurat
        merged = df1.merge(df2, on=id_col, suffixes=('_1', '_2'), how='inner')
        
        # Hitung perbedaan
        pred_col_1 = f"{pred_col}_1"
        pred_col_2 = f"{pred_col}_2"
        
        if pred_col_1 not in merged.columns:
            pred_col_1 = pred_col
        if pred_col_2 not in merged.columns:
            pred_col_2 = pred_col
        
        differences = merged[pred_col_1] != merged[pred_col_2]
        n_different = differences.sum()
        n_total = len(merged)
        percentage = (n_different / n_total) * 100 if n_total > 0 else 0
        
        print("\n" + "-"*80)
        print("PREDICTION COMPARISON")
        print("-"*80)
        print(f"Total rows dibandingkan: {n_total}")
        print(f"Rows yang SAMA: {n_total - n_different} ({100 - percentage:.2f}%)")
        print(f"Rows yang BERBEDA: {n_different} ({percentage:.2f}%)")
        print("-"*80)
        
        # Tampilkan detail perbedaan jika tidak terlalu banyak
        if n_different > 0 and n_different <= 20:
            print("\nDETAIL PERBEDAAN:")
            diff_rows = merged[differences][[id_col, pred_col_1, pred_col_2]]
            print(diff_rows.to_string(index=False))
        elif n_different > 20:
            print(f"\nTerlalu banyak perbedaan untuk ditampilkan ({n_different} rows)")
            print("Sample 10 perbedaan pertama:")
            diff_rows = merged[differences][[id_col, pred_col_1, pred_col_2]].head(10)
            print(diff_rows.to_string(index=False))
        
        # Statistik prediksi per kelas
        print("\n" + "-"*80)
        print("DISTRIBUTION COMPARISON")
        print("-"*80)
        
        print(f"\nFile 1 - {pred_col}:")
        print(df1[pred_col].value_counts().sort_index())
        
        print(f"\nFile 2 - {pred_col}:")
        print(df2[pred_col].value_counts().sort_index())
        
        print("\n" + "="*80)
        
        return {
            'total_rows': n_total,
            'same': n_total - n_different,
            'different': n_different,
            'percentage_different': percentage,
            'percentage_same': 100 - percentage
        }
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: File tidak ditemukan - {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function untuk menjalankan script."""
    
    if len(sys.argv) == 3:
        # Path diberikan sebagai command line arguments
        file1_path = sys.argv[1]
        file2_path = sys.argv[2]
    else:
        # Minta input dari user
        print("COMPARE SUBMISSIONS")
        print("="*80)
        file1_path = input("Masukkan path file submission 1: ").strip().strip('"')
        file2_path = input("Masukkan path file submission 2: ").strip().strip('"')
    
    # Validasi file exists
    if not os.path.exists(file1_path):
        print(f"❌ ERROR: File tidak ditemukan: {file1_path}")
        return
    
    if not os.path.exists(file2_path):
        print(f"❌ ERROR: File tidak ditemukan: {file2_path}")
        return
    
    # Lakukan perbandingan
    results = compare_submissions(file1_path, file2_path)
    
    if results:
        print("\n✓ Perbandingan selesai!")
        return results


if __name__ == "__main__":
    main()
