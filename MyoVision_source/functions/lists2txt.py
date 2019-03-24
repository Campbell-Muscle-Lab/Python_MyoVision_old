import scipy as sp


# START lists2txt**************************************************************
# Description:
#     takes in ndarray arrays and a set of corresponding headers and puts them
#     into a text file
# Inputs:
#     *lists: any number of 1-dimensional ndarrays of the same length
#     headers: a 1-dimensional ndarray of strings containing header names
#     filename: full path of output .txt file
# Outputs:
#     N/A
def lists2txt(*lists, headers, filename):
    # Start by checking that # headings = # of lists
    if (not lists):
        print("No lists passed in.\nOutput file not changed.")
    else:
        num_lists = len(lists)
        list_length = lists[0].size

        # Adds default headers if not enough were passed in
        if (headers.size < num_lists):
            print("Number of headers less than number of lists. Adding default headers.")
            default_number = 0;
            while (headers.size < num_lists):
                def_head = "Header %s"%default_number
                headers = sp.append(headers, def_head)
                default_number = default_number+1;

        # Removes headers from end of the list if too many were passed in
        elif (headers.size > num_lists):
            print("Too many header names passed in. Truncating list.")
            idx = range(len(lists),(headers.size))
            headers = sp.delete(headers, idx)

        # Creates header line for output txt file
        pad_num = 20
        dec_trunc = 4
        header_line = ""
        # Makes header line
        for j in range(headers.size):
            header_line = header_line + "{}".format(headers[j])
            if (j != num_lists-1):
                header_line = header_line + "\t"
            # header_line = header_line + "{:<{num}}".format(headers[j], num=pad_num)
        header_line = header_line + "\n"

        # Creates and appends header line and list elements to output txt file
        if '.txt' not in filename:
            filename = filename + '.txt'
        with open(filename, 'w') as f:
            f.write(header_line)
            for l in range(list_length):
                line = ""
                for i in range(num_lists):
                    line = line + "{0:.{trunc}f}".format(lists[i][l], trunc=dec_trunc)
                    if (i != num_lists-1):
                        line = line + "\t"
                    # line = line + "{:<{num}}".format(lists[i][l], num=pad_num)
                line = line + "\n"
                f.write(line)
            f.close()
# END lists2txt////////////////////////////////////////////////////////////////