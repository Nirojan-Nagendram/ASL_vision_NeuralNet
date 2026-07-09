# This script has all my universal functions needed in most my programmes.

def logger(label, value, prev_value = None, debug = False, gap = False,save_file = None):
    if not debug:
        return
    should_print = prev_value is None or value != prev_value      # combine conditions 
    output = f"{label} : {value}"
    if gap:
        output += "\n"
    if should_print:
        print(output)
    if save_file:
        pass    #save file as csv?

# Logger debug
value = (3,4,5)
i = 0
while i<5:
    i +=1
    logger("label",value, debug=False)