# Logo: ~180px wide
html_parts.append(
    f'<p style="margin:0; text-align:left;">'
    f'<img src="cid:logo_{identifier_slug}" '
    f'width="180" height="60" '
    f'style="width:180px; height:60px; display:block;">'
    f'</p>'
)

# Chart: ~550px wide
html_parts.append(
    f'<table width="600" cellpadding="0" cellspacing="0" border="0">'
    f'  <tr>'
    f'    <td width="600" align="left">'
    f'      <img src="cid:image{cid_counter}" '
    f'           width="550" '
    f'           style="width:550px; height:auto; display:block;">'
    f'    </td>'
    f'  </tr>'
    f'</table>'
)
