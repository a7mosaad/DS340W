import os
import sys
import email
from email import policy
from email.parser import BytesParser
from pathlib import Path
import tempfile
import shutil

try:
    from xhtml2pdf import pisa
    from bs4 import BeautifulSoup
    import extract_msg
    from PIL import Image
except ImportError as e:
    print(f"Missing required library: {e}")
    print("\nInstall: pip install xhtml2pdf extract-msg pillow beautifulsoup4")
    sys.exit(1)


class EmailToPDFConverter:
    def __init__(self, email_path, output_path=None):
        self.email_path = Path(email_path)
        if not self.email_path.exists():
            raise FileNotFoundError(f"Email file not found: {email_path}")
        self.output_path = Path(output_path) if output_path else self.email_path.with_suffix('.pdf')
        self.temp_dir = None
        self.embedded_images = {}
        self.attachments = []

    def convert(self):
        print(f"Converting {self.email_path} to PDF...")
        self.temp_dir = Path(tempfile.mkdtemp())
        try:
            if self.email_path.suffix.lower() == '.msg':
                html_content = self._parse_msg()
            elif self.email_path.suffix.lower() == '.eml':
                html_content = self._parse_eml()
            else:
                raise ValueError(f"Unsupported file format: {self.email_path.suffix}")
            self._html_to_pdf(html_content)
            print(f"✓ Successfully converted to: {self.output_path}")
        finally:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def _parse_eml(self):
        with open(self.email_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        subject = msg.get('Subject', 'No Subject')
        from_addr = msg.get('From', 'Unknown')
        to_addr = msg.get('To', 'Unknown')
        date = msg.get('Date', 'Unknown')
        cc_addr = msg.get('Cc', '')
        html_body = None
        text_body = None
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                if content_type == 'text/html' and 'attachment' not in content_disposition:
                    html_body = part.get_content()
                elif content_type == 'text/plain' and 'attachment' not in content_disposition:
                    text_body = part.get_content()
                elif content_type.startswith('image/'):
                    self._extract_embedded_image(part)
                elif 'attachment' in content_disposition or part.get_filename():
                    self._extract_attachment(part)
        else:
            if msg.get_content_type() == 'text/html':
                html_body = msg.get_content()
            else:
                text_body = msg.get_content()
        return self._build_html(subject, from_addr, to_addr, date, cc_addr, html_body, text_body)

    def _parse_msg(self):
        msg = extract_msg.Message(str(self.email_path))
        subject = msg.subject or 'No Subject'
        from_addr = msg.sender or 'Unknown'
        to_addr = msg.to or 'Unknown'
        date = msg.date or 'Unknown'
        cc_addr = msg.cc or ''
        html_body = msg.htmlBody
        text_body = msg.body
        for attachment in msg.attachments:
            att_name = attachment.longFilename or attachment.shortFilename
            att_data = attachment.data
            if att_data:
                self.attachments.append({'name': att_name, 'size': len(att_data)})
                att_path = self.temp_dir / att_name
                with open(att_path, 'wb') as f:
                    f.write(att_data)
        msg.close()
        return self._build_html(subject, from_addr, to_addr, str(date), cc_addr, html_body, text_body)

    def _extract_embedded_image(self, part):
        content_id = part.get('Content-ID', '').strip('<>')
        filename = part.get_filename()
        if not filename:
            ext = part.get_content_subtype()
            filename = f"{content_id or 'image'}.{ext}"
        image_path = self.temp_dir / filename
        with open(image_path, 'wb') as f:
            f.write(part.get_payload(decode=True))
        if content_id:
            self.embedded_images[f'cid:{content_id}'] = str(image_path)

    def _extract_attachment(self, part):
        filename = part.get_filename()
        if filename:
            data = part.get_payload(decode=True)
            self.attachments.append({'name': filename, 'size': len(data)})
            with open(self.temp_dir / filename, 'wb') as f:
                f.write(data)

    def _process_html_body(self, html_body):
        soup = BeautifulSoup(html_body, 'html.parser')
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src.startswith('cid:') and src in self.embedded_images:
                img['src'] = f'file://{self.embedded_images[src]}'
        for table in soup.find_all('table'):
            current_style = table.get('style', '')
            if 'border-collapse' not in current_style:
                table['style'] = current_style + '; border-collapse: collapse;'
            for cell in table.find_all(['td', 'th']):
                if cell.get('bgcolor'):
                    bgcolor = cell['bgcolor']
                    current_style = cell.get('style', '')
                    if 'background-color' not in current_style:
                        cell['style'] = current_style + f'; background-color: {bgcolor};'
                current_style = cell.get('style', '')
                if 'border' not in current_style:
                    cell['style'] = current_style + '; border: 1px solid #000;'
        return str(soup)

    def _build_html(self, subject, from_addr, to_addr, date, cc_addr, html_body, text_body):
        css = """
        <style>
            @page { margin: 1cm; }
            body { font-family: Arial, sans-serif; font-size: 11pt; color: #333; padding: 20px; }
            .email-subject { font-size: 14pt; font-weight: bold; margin-bottom: 15px; }
            .email-header { background-color: #f5f5f5; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; font-size: 10pt; }
            .email-header-row { margin: 5px 0; }
            .email-header-label { font-weight: bold; display: inline-block; width: 80px; color: #555; }
            .email-body { border-top: 2px solid #ddd; padding-top: 20px; }
            table { border-collapse: collapse; margin: 10px 0; font-size: 10pt; }
            td, th { border: 1px solid #000; padding: 5px 8px; vertical-align: top; }
            img { max-width: 100%; }
            pre { background-color: #f5f5f5; padding: 10px; white-space: pre-wrap; }
            .attachments { margin-top: 30px; padding: 15px; background-color: #f9f9f9; border: 1px solid #ddd; }
        </style>
        """

        def esc(text):
            return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        cc_row = f'<div class="email-header-row"><span class="email-header-label">Cc:</span><span>{esc(cc_addr)}</span></div>' if cc_addr else ''

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">{css}</head><body>
<div class="email-subject">{esc(subject)}</div>
<div class="email-header">
  <div class="email-header-row"><span class="email-header-label">From:</span><span>{esc(from_addr)}</span></div>
  <div class="email-header-row"><span class="email-header-label">To:</span><span>{esc(to_addr)}</span></div>
  {cc_row}
  <div class="email-header-row"><span class="email-header-label">Date:</span><span>{esc(date)}</span></div>
</div>
<div class="email-body">
"""
        if html_body:
            html += self._process_html_body(html_body)
        elif text_body:
            html += f'<pre>{text_body}</pre>'
        else:
            html += '<p><em>No content</em></p>'

        html += '</div>\n'

        if self.attachments:
            html += '<div class="attachments"><strong>Attachments:</strong><br/>'
            for att in self.attachments:
                html += f'📎 {esc(att["name"])} ({att["size"] / 1024:.1f} KB)<br/>'
            html += '</div>'

        html += '</body></html>'
        return html

    def _html_to_pdf(self, html_content):
        # Save debug HTML
        debug_path = self.temp_dir / 'debug.html'
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        with open(self.output_path, 'wb') as pdf_file:
            pisa_status = pisa.CreatePDF(
                html_content,
                dest=pdf_file,
                encoding='utf-8'
            )
        if pisa_status.err:
            raise Exception(f"PDF conversion failed with {pisa_status.err} errors")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert .eml or .msg email to PDF')
    parser.add_argument('email_file', help='Path to the email file')
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
