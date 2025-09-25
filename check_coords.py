for filename in ['deck_twist_000.html', 'deck_twist_045.html', 'deck_twist_090.html', 'deck_section_rot_000.html', 'deck_section_rot_030.html', 'deck_section_rot_090.html']:
    print(f"\n=== {filename} ===")
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        start = content.find('Cross-section @ 0.000 m')
        if start != -1:
            data_start = content.find('"x":', start)
            if data_start != -1:
                x_start = content.find('[', data_start)
                x_end = content.find(']', x_start) + 1
                x_data = content[x_start:x_end]
                print('X data sample:', x_data[:50])
                
                y_start = content.find('"y":', x_end)
                y_start = content.find('[', y_start)
                y_end = content.find(']', y_start) + 1
                y_data = content[y_start:y_end]
                print('Y data sample:', y_data[:50])
                
                z_start = content.find('"z":', y_end)
                z_start = content.find('[', z_start)
                z_end = content.find(']', z_start) + 1
                z_data = content[z_start:z_end]
                print('Z data sample:', z_data[:50])