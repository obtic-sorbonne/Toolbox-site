from pathlib import Path
import subprocess
import glob
from io import StringIO
import re
import os
from werkzeug.utils import secure_filename

def ensure_dir(path):
    """Ensure directory exists and has correct permissions"""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(str(path), 0o777)
    return path

def check_tesseract_setup():
    """Verify Tesseract installation and available languages"""
    try:
        # Check Tesseract version
        version = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        print("Tesseract version:")
        print(version.stdout)
        
        # Check TESSDATA_PREFIX
        tessdata = os.getenv('TESSDATA_PREFIX')
        print(f"TESSDATA_PREFIX: {tessdata}")
        if tessdata:
            print(f"TESSDATA contents: {os.listdir(tessdata) if os.path.exists(tessdata) else 'Directory not found'}")
            
        # List available languages
        langs = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True)
        #print("Available languages:")
        #print(langs.stdout)
        
        return True
    except Exception as e:
        print(f"Tesseract setup check failed: {e}")
        return False

def tesseract_to_txt(uploaded_files, model, model_bis, rand_name, ROOT_FOLDER, UPLOAD_FOLDER):
    try:
        # Debug information
        print(f"\n=== Starting OCR process ===")
        print(f"Model: {model}")
        print(f"Model bis: {model_bis}")
        
        # Verify Tesseract setup
        if not check_tesseract_setup():
            raise Exception("Tesseract verification failed")

        # Create base paths
        root_path = Path(ROOT_FOLDER)
        upload_path = root_path / UPLOAD_FOLDER
        result_path = ensure_dir(upload_path / rand_name)
        
        # Extensions supported
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        
        output_stream = StringIO()
        print(f"Result path: {result_path}")
        
        if model_bis:
            model = f"{model}+{model_bis}"
            
        for f in uploaded_files:
            filename = secure_filename(f.filename)
            file_path = Path(filename)
            file_stem = file_path.stem
            file_extension = file_path.suffix.lower()
            
            print(f"\n=== Processing file: {filename} ===")
            print(f"Extension: {file_extension}")
            
            if file_extension == '.pdf':
                # Create temp directory for PDF processing
                temp_dir = ensure_dir(upload_path / f"{file_stem}_temp")
                pdf_path = temp_dir / filename
                
                # Save PDF
                f.save(str(pdf_path))
                print(f"Saved PDF to: {pdf_path}")
                
                # Convert to PNG
                try:
                    print("Converting PDF to PNG...")
                    result = subprocess.run([
                        'pdftoppm', '-r', '180',
                        str(pdf_path),
                        str(temp_dir / file_stem),
                        '-png'
                    ], capture_output=True, text=True, check=True)
                    print("PDF conversion output:", result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"PDF conversion error output: {e.stderr}")
                    raise Exception(f"PDF conversion failed: {e}")
                
                # Process each PNG
                png_files = sorted(
                    glob.glob(str(temp_dir / '*.png')),
                    key=lambda f: int(re.sub(r'\D', '', f)) if re.search(r'\d+', f) else 0
                )
                
                print(f"Found {len(png_files)} PNG files to process")
                for png_file in png_files:
                    png_path = Path(png_file)
                    output_base = png_path.with_suffix('')
                    
                    try:
                        print(f"\nProcessing PNG: {png_path}")
                        result = subprocess.run([
                            'tesseract',
                            '-l', model,
                            str(png_path),
                            str(output_base)
                        ], capture_output=True, text=True, check=True)
                        print("Tesseract output:", result.stdout)
                        
                        txt_path = output_base.with_suffix('.txt')
                        if not txt_path.exists():
                            raise FileNotFoundError(f"Tesseract output not found: {txt_path}")
                            
                        with txt_path.open('r', encoding='utf-8') as ftxt:
                            output_stream.write(ftxt.read())
                            output_stream.write('\n\n')
                            
                    except subprocess.CalledProcessError as e:
                        print(f"Tesseract error output: {e.stderr}")
                        raise Exception(f"OCR failed for {png_path}: {e}")
                            
            elif file_extension in extensions:
                # Process single image
                image_path = result_path / filename
                output_base = result_path / file_stem
                
                # Save image
                f.save(str(image_path))
                print(f"Saved image to: {image_path}")
                
                try:
                    print("Running Tesseract OCR...")
                    result = subprocess.run([
                        'tesseract',
                        '-l', model,
                        str(image_path),
                        str(output_base)
                    ], capture_output=True, text=True, check=True)
                    print("Tesseract output:", result.stdout)
                    
                    txt_path = output_base.with_suffix('.txt')
                    if not txt_path.exists():
                        raise FileNotFoundError(f"Tesseract output not found: {txt_path}")
                        
                    with txt_path.open('r', encoding='utf-8') as ftxt:
                        output_stream.write(ftxt.read())
                        output_stream.write('\n\n')
                        
                except subprocess.CalledProcessError as e:
                    print(f"Tesseract error output: {e.stderr}")
                    raise Exception(f"OCR failed for {filename}: {e}")
            
            else:
                raise Exception(f"Unsupported file extension: {file_extension}")
        
        final_text = output_stream.getvalue()
        output_stream.close()
        return final_text
        
    except Exception as e:
        print(f"Error in tesseract_to_txt: {e}")
        raise
