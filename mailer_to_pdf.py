"""
Email to PDF Converter - Enhanced for Inline Excel Tables
Converts email files (.eml or .msg) to PDF while preserving:
- HTML formatting and colors
- Embedded images (PNGs, JPEGs, etc.)
- Excel tables pasted directly into email (with cell colors, borders, formatting)
- Text content
- Email headers (From, To, Subject, Date)
"""

import os
import sys
import email
from email import policy
from email.parser import BytesParser
import base64
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import re
from xhtml2pdf import pisa

# Required libraries
try:
    from weasyprint import HTML, CSS
    from bs4 import BeautifulSoup
    import extract_msg
    from PIL import Image
except ImportError as e:
    print(f"Missing required library: {e}")
    print("\nInstall required packages with:")
    print("pip install weasyprint extract-msg pillow beautifulsoup4")
    sys.exit(1)


class EmailToPDFConverter:
    def __init__(self, email_path, output_path=None):
        """
        Initialize the converter
        
        Args:
            email_path (str): Path to the email file (.eml or .msg)
            output_path (str): Path for the output PDF (optional)
        """
        self.email_path = Path(email_path)
        
        if not self.email_path.exists():
            raise FileNotFoundError(f"Email file not found: {email_path}")
        
        # Set output path
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.email_path.with_suffix('.pdf')
        
        self.temp_dir = None
        self.embedded_images = {}
        self.attachments = []
    
    def convert(self):
        """Main conversion method"""
        print(f"Converting {self.email_path} to PDF...")
        
        # Create temporary directory for processing
        self.temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Parse email based on file type
            if self.email_path.suffix.lower() == '.msg':
                html_content = self._parse_msg()
            elif self.email_path.suffix.lower() == '.eml':
                html_content = self._parse_eml()
            else:
                raise ValueError(f"Unsupported file format: {self.email_path.suffix}")
            
            # Convert HTML to PDF
            self._html_to_pdf(html_content)
            
            print(f"✓ Successfully converted to: {self.output_path}")
            
            if self.attachments:
                print(f"\nFound {len(self.attachments)} attachment(s):")
                for att in self.attachments:
                    print(f"  - {att['name']} ({att['size']} bytes)")
        
        finally:
            # Clean up temporary directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def _parse_eml(self):
        """Parse .eml email file"""
        with open(self.email_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        # Extract email metadata
        subject = msg.get('Subject', 'No Subject')
        from_addr = msg.get('From', 'Unknown')
        to_addr = msg.get('To', 'Unknown')
        date = msg.get('Date', 'Unknown')
        cc_addr = msg.get('Cc', '')
        
        # Get email body
        html_body = None
        text_body = None
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Extract HTML body
                if content_type == 'text/html' and 'attachment' not in content_disposition:
                    html_body = part.get_content()
                
                # Extract plain text body
                elif content_type == 'text/plain' and 'attachment' not in content_disposition:
                    text_body = part.get_content()
                
                # Extract embedded images
                elif content_type.startswith('image/'):
                    self._extract_embedded_image(part)
                
                # Extract attachments
                elif 'attachment' in content_disposition or part.get_filename():
                    self._extract_attachment(part)
        else:
            if msg.get_content_type() == 'text/html':
                html_body = msg.get_content()
            else:
                text_body = msg.get_content()
        
        # Build HTML content
        return self._build_html(subject, from_addr, to_addr, date, cc_addr, 
                               html_body, text_body)
    
    def _parse_msg(self):
        """Parse .msg Outlook email file"""
        msg = extract_msg.Message(str(self.email_path))
        
        # Extract metadata
        subject = msg.subject or 'No Subject'
        from_addr = msg.sender or 'Unknown'
        to_addr = msg.to or 'Unknown'
        date = msg.date or 'Unknown'
        cc_addr = msg.cc or ''
        
        # Get body
        html_body = msg.htmlBody
        text_body = msg.body
        
        # Extract attachments
        for attachment in msg.attachments:
            att_data = {
                'name': attachment.longFilename or attachment.shortFilename,
                'data': attachment.data,
                'size': len(attachment.data) if attachment.data else 0
            }
            self.attachments.append(att_data)
            
            # Save attachment to temp directory
            if att_data['data']:
                att_path = self.temp_dir / att_data['name']
                with open(att_path, 'wb') as f:
                    f.write(att_data['data'])
        
        msg.close()
        
        return self._build_html(subject, from_addr, to_addr, str(date), cc_addr,
                               html_body, text_body)
    
    def _extract_embedded_image(self, part):
        """Extract and save embedded images"""
        content_id = part.get('Content-ID', '')
        if content_id:
            content_id = content_id.strip('<>')
        
        filename = part.get_filename()
        if not filename:
            # Generate filename from content-id or use default
            ext = part.get_content_subtype()
            filename = f"{content_id or 'image'}.{ext}"
        
        # Save image to temp directory
        image_path = self.temp_dir / filename
        with open(image_path, 'wb') as f:
            f.write(part.get_payload(decode=True))
        
        # Store mapping for CID references
        if content_id:
            self.embedded_images[f'cid:{content_id}'] = str(image_path)
        
        return str(image_path)
    
    def _extract_attachment(self, part):
        """Extract email attachments"""
        filename = part.get_filename()
        if filename:
            att_data = {
                'name': filename,
                'data': part.get_payload(decode=True),
                'size': len(part.get_payload(decode=True))
            }
            self.attachments.append(att_data)
            
            # Save to temp directory
            att_path = self.temp_dir / filename
            with open(att_path, 'wb') as f:
                f.write(att_data['data'])
    
    def _build_html(self, subject, from_addr, to_addr, date, cc_addr, 
                   html_body, text_body):
        """Build complete HTML for PDF conversion"""
        
        # CSS styling - Enhanced for Excel table preservation
        css = """
        <style>
            @page {
                margin: 1cm;
                size: A4;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #333;
                margin: 0;
                padding: 20px;
            }
            
            .email-header {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                font-size: 10pt;
            }
            
            .email-header-row {
                margin: 5px 0;
            }
            
            .email-header-label {
                font-weight: bold;
                display: inline-block;
                width: 80px;
                color: #555;
            }
            
            .email-subject {
                font-size: 14pt;
                font-weight: bold;
                margin-bottom: 15px;
                color: #000;
            }
            
            .email-body {
                border-top: 2px solid #ddd;
                padding-top: 20px;
            }
            
            /* Preserve Excel table styling */
            table {
                border-collapse: collapse;
                margin: 10px 0;
                font-size: 10pt;
                page-break-inside: auto;
            }
            
            table tr {
                page-break-inside: avoid;
                page-break-after: auto;
            }
            
            table td, table th {
                padding: 5px 8px;
                border: 1px solid #000;
                vertical-align: top;
                word-wrap: break-word;
            }
            
            /* Preserve inline styles from Excel */
            table[style] {
                /* Keep all inline table styles */
            }
            
            td[style], th[style] {
                /* Keep all inline cell styles */
            }
            
            /* Handle Excel's typical cell formatting */
            .xl-cell {
                white-space: nowrap;
            }
            
            .attachments {
                margin-top: 30px;
                padding: 15px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            
            .attachment-title {
                font-weight: bold;
                margin-bottom: 10px;
                color: #555;
            }
            
            .attachment-item {
                padding: 5px 0;
            }
            
            img {
                max-width: 100%;
                height: auto;
            }
            
            pre {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 3px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            /* Preserve background colors from Excel */
            td[bgcolor], th[bgcolor] {
                /* Colors preserved via inline bgcolor attribute */
            }
            
            /* Font colors */
            font[color] {
                /* Colors preserved via inline font color */
            }
        </style>
        """
        
        # Build HTML header section
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            {css}
        </head>
        <body>
            <div class="email-subject">{self._escape_html(subject)}</div>
            
            <div class="email-header">
                <div class="email-header-row">
                    <span class="email-header-label">From:</span>
                    <span>{self._escape_html(from_addr)}</span>
                </div>
                <div class="email-header-row">
                    <span class="email-header-label">To:</span>
                    <span>{self._escape_html(to_addr)}</span>
                </div>
        """
        
        if cc_addr:
            html += f"""
                <div class="email-header-row">
                    <span class="email-header-label">Cc:</span>
                    <span>{self._escape_html(cc_addr)}</span>
                </div>
            """
        
        html += f"""
                <div class="email-header-row">
                    <span class="email-header-label">Date:</span>
                    <span>{self._escape_html(str(date))}</span>
                </div>
            </div>
            
            <div class="email-body">
        """
        
        # Add email body
        if html_body:
            # Process HTML body - preserve inline Excel table formatting
            processed_html = self._process_html_body(html_body)
            html += processed_html
        elif text_body:
            # Convert plain text to HTML
            html += f'<pre>{self._escape_html(text_body)}</pre>'
        else:
            html += '<p><em>No content</em></p>'
        
        html += '</div>\n'
        
        # Add attachments list if any
        if self.attachments:
            html += """
            <div class="attachments">
                <div class="attachment-title">Attachments:</div>
            """
            
            for att in self.attachments:
                size_kb = att['size'] / 1024
                html += f'<div class="attachment-item">📎 {self._escape_html(att["name"])} ({size_kb:.1f} KB)</div>\n'
            
            html += '</div>\n'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _process_html_body(self, html_body):
        """
        Process HTML body to preserve Excel table formatting and embedded images
        
        Excel tables pasted into emails typically have:
        - Inline style attributes (background-color, color, font-weight, etc.)
        - Border attributes
        - Cell padding and spacing
        - Font color/size via <font> tags or CSS
        """
        soup = BeautifulSoup(html_body, 'html.parser')
        
        # Replace CID references with local file paths for embedded images
        for img in soup.find_all('img'):
            src = img.get('src', '')
            
            if src.startswith('cid:'):
                if src in self.embedded_images:
                    # Convert to file:// URL for weasyprint
                    img['src'] = f'file://{self.embedded_images[src]}'
            
            # Handle base64 embedded images (keep as-is)
            elif src.startswith('data:image'):
                pass
            
            # Handle external URLs (keep as-is)
            elif src.startswith('http'):
                pass
        
        # Process tables to preserve Excel formatting
        for table in soup.find_all('table'):
            # Preserve all table attributes (style, border, cellpadding, etc.)
            # Excel tables often have these attributes set
            
            # Ensure border-collapse is preserved
            current_style = table.get('style', '')
            if 'border-collapse' not in current_style:
                # Add border-collapse if not present
                if current_style and not current_style.endswith(';'):
                    current_style += ';'
                table['style'] = current_style + ' border-collapse: collapse;'
            
            # Process all cells in the table
            for cell in table.find_all(['td', 'th']):
                # Preserve all inline styles (background-color, color, font-weight, etc.)
                # These are already in the cell's 'style' attribute
                
                # Handle bgcolor attribute (common in Excel-generated HTML)
                if cell.get('bgcolor'):
                    bgcolor = cell['bgcolor']
                    current_style = cell.get('style', '')
                    if 'background-color' not in current_style:
                        if current_style and not current_style.endswith(';'):
                            current_style += ';'
                        cell['style'] = current_style + f' background-color: {bgcolor};'
                
                # Preserve font tags (Excel often uses these for colors)
                # BeautifulSoup will keep them as-is
                
                # Ensure cell borders are visible
                current_style = cell.get('style', '')
                if 'border' not in current_style:
                    if current_style and not current_style.endswith(';'):
                        current_style += ';'
                    cell['style'] = current_style + ' border: 1px solid #000;'
        
        # Convert back to HTML string
        # Use formatter=None to preserve all attributes exactly as they are
        return str(soup)
    
    def _escape_html(self, text):
        """Escape HTML special characters"""
        if not isinstance(text, str):
            text = str(text)
        
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
    
    def _html_to_pdf(self, html_content):
        from xhtml2pdf import pisa

        with open(self.output_path, 'wb') as pdf_file:
            pisa_status = pisa.CreatePDF(
                html_content,
                dest=pdf_file,
                encoding='utf-8'
            )

    if pisa_status.err:
        raise Exception(f"PDF conversion failed with {pisa_status.err} errors")


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert email files (.eml or .msg) to PDF with preserved formatting and inline Excel tables'
    )
    parser.add_argument('email_file', help='Path to the email file (.eml or .msg)')
    parser.add_argument('-o', '--output', help='Output PDF path (optional)')
    
    args = parser.parse_args()
    
    try:
        converter = EmailToPDFConverter(args.email_file, args.output)
        converter.convert()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
