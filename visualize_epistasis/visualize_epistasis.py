

from pymol import cmd
import os

def read_csv(file, delim=','):
    if file is None or not os.path.exists(file):
        print('Error: no data file given, or the path is incorrect.')
        return None, None
    pairings, scores = [], []
    with open(file, 'r') as f:
        header_line = next(f) # skip header
        next(f) # skip intercept
        for line in f:
            sep = line.split(',')
            # expect the first column to give atom pairs and the second column to give a number
            pairings.append( sep[0].strip().split('|') )
            scores.append(float(sep[1].strip()))
    # print(pairings)
    return pairings, scores

def centroid(coords):
    n = len(coords)
    xyz = [0,0,0]
    for i in range(0,3):
        mean = 0
        for rec in coords:
            mean += rec[i]
        xyz[i] = mean / n
    return xyz

def visualize_epistasis(file, obj, max_atom_scale=2, min_atom_scale=0.5, max_line_scale=2, min_line_scale=0.5, pos_color=[1,0,0], neg_color=[0,0,1], backbone_color=[0.8,0.8,0.8]):
    '''
        Inputs
        file - data file of epistatic effects in csv format.
        obj - model object to visualize on. Usually the structure has the same name as the pbd identifier.
        max_atom_scale - maximum scale to size atoms by. Maximum size is assigned to the largest absolute value of epistatic effects.
        min_atom_scale - minimum scale to size atoms by. Minimum size is assigned to the smallest absolute value of epistatic effects.
        max_line_scale - maximum scale to size lines by. Maximum size is assigned to the largest absolute value of epistatic effects.
        min_line_scale - minimum scale to size lines by. Minimum size is assigned to the smallest absolute value of epistatic effects.
        pos_color - color for positive effects.
        neg_color - color for negative effects.
        backbone_color - color for the cartoon representation of the structure.
    '''
    
    # fetch the current vwd radius
    vdw_space = {'vdw_list':[]}
    cmd.iterate_state( -1, "name CA", "vdw_list.append(vdw)" , space=vdw_space)
    CA_vdw = vdw_space['vdw_list'][0]
    
    max_atom_scale = float(max_atom_scale) 
    min_atom_scale = float(min_atom_scale)
    max_line_scale = float(max_line_scale)
    min_line_scale = float(min_line_scale)
    # house keeping
    # make sure the file exists
    p, s = read_csv(file)
    if p is None or s is None:
        print('Error: plotting failed, invalid data file.')
        return
    # make sure the chains exists
    chains = cmd.get_chains(obj)
    
    # program in colors
    cmd.set_color("pos_color", pos_color)
    cmd.set_color("neg_color", neg_color)
    
    # make sure colors do not smear
    cmd.set("cartoon_discrete_colors", "on")
    
    # determine order of epistasis, first order is plotted on atoms while higher levels are lines connecting atoms
    atom = [ True if len(x)<=1 else False for x in p ]
    
    # normalize scores
    abs_score = [ abs(x) for x in s ]
    upper = max(abs_score)
    lower = min(abs_score)
    score_range = upper-lower
    
    # calculate scaling variables
    atom_scale_range = float(max_atom_scale)-float(min_atom_scale)
    line_scale_range = float(max_line_scale)-float(min_line_scale)
    scales = [ (abs_score[i]-lower) / score_range for i in range(len(abs_score)) ]
    # set to atom or line scaling, respectively
    scales = [ scales[i]*atom_scale_range+min_atom_scale if atom[i] else scales[i]*line_scale_range+min_line_scale for i in range(len(scales)) ]
    
    # do some plotting
    
    sel_temp = '{obj}//{chain}/{resi}/CA' # selection template
    for chain in chains:
        sel = sel_temp.format(obj=obj, chain=chain, resi="{resi}")
        for i in range(len(p)):
            scale = scales[i]
            # decide color
            score = s[i]
            
            color = "pos_color"
            if score < 0:
                color = "neg_color"
            # reset name   
            name = "" 
            # visualize
            if atom[i]: # first order
                name=p[i][0]
                cmd.select( name, sel.format( resi=p[i][0]) )
                cmd.show("sphere", name)
                cmd.set("sphere_scale", scale/CA_vdw, name)
            elif len(p[i])==2: # second order
                name="{}_{}".format(p[i][0], p[i][1])
                cmd.distance( name, sel.format(resi=p[i][0]), sel.format(resi=p[i][1]) )
                cmd.set("dash_radius", scale, name)
            elif len(p[i])>2: # higher order
                # get all linked atoms and find the centroid
                target_resi = ""
                
                for j in range(len(p[i])):
                    if j==0:
                        target_resi = p[i][j]
                        name = p[i][j]
                    else:
                        target_resi = target_resi + "+" + p[i][j]
                        name = name + "_" + p[i][j]
                group_sel = sel.format(resi=target_resi) # make selection
                coords = []
                myspace = {'coords':coords}
                cmd.iterate_state( -1, group_sel, "coords.append([x,y,z])" , space=myspace ) # get list of xyz coordinates
                xyz = centroid(coords) # calculate the centroid
                # create a pseudoatom at the centroid, and link all atoms to it with lines
                cmd.pseudoatom( "pseudoatoms", name=name, chain=chain, pos=xyz)
                for x in p[i]:
                    cmd.distance(name, "pseudoatoms//{}//{}".format(chain,name), sel.format(resi=x) )
                cmd.set("dash_radius", scale, name)
            # clean up after each set of data
            cmd.color(color, name)
    # final clean up
    cmd.set("dash_gap",0)
    cmd.hide("everything","pseudoatoms")
    cmd.hide('labels')        
    cmd.set('cartoon_color', backbone_color)
                

cmd.extend('visualize_epistasis', visualize_epistasis)