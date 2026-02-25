import sys

def convert_to_imod(aln_path,output_path):
    with open(aln_path, 'r') as f:
        lines = f.readlines()
    parts = lines[1].strip().split()

    from cryoet_alignment import read
    from cryoet_alignment import write
    from cryoet_alignment.io.cryoet_data_portal import Alignment
    aretomo3_alignment = read(aln_path)
    aln = Alignment.from_aretomo3(aretomo3_alignment)
    tilt_series_dim = (int(parts[3]), int(parts[4]), int(parts[5]))
    write(aln.to_imod(ts_size=tilt_series_dim), output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_aln_to_xf.py input.aln output.xf")
        sys.exit(1)

    aln_file = sys.argv[1]
    xf_file = sys.argv[2]
    convert_to_imod(aln_file, xf_file)
