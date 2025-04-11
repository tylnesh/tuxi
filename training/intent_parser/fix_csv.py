import csv
import sys

def fix_csv(input_file, output_file):
    """
    Reads the CSV file from input_file, and writes a new CSV to output_file.
    If a row contains more than two columns, all columns except the last one are joined as the 'text' field.
    The last column is assumed to be the 'label' field.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            # Skip empty lines
            if not row:
                continue

            # If row is already well-formed (2 fields), write as is.
            if len(row) == 2:
                writer.writerow(row)
            # If row has more than two fields, join the extra fields into one text field.
            elif len(row) > 2:
                # Join all fields except the last one to form a fixed text entry.
                fixed_text = ','.join(row[:-1]).strip()
                fixed_label = row[-1].strip()
                writer.writerow([fixed_text, fixed_label])
            else:
                # In case a row has fewer than 2 fields, write it unchanged or handle it as needed.
                writer.writerow(row)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_csv.py input_file.csv [output_file.csv]")
    else:
        input_file = sys.argv[1]
        # Default output filename: "fixed_" + input filename if not provided.
        output_file = sys.argv[2] if len(sys.argv) > 2 else "fixed_" + input_file
        fix_csv(input_file, output_file)
        print(f"Fixed CSV file saved as {output_file}")
