import numpy as np 
import math
from tqdm import tqdm 
from .utils import *
import os

class Attack_idealW:
    def __init__(self, params,i=None):
        self.params = params
        self.process = []
        self.i = i 

    def completion_procedure(self, success, x_adv, queries, loc, patch, loss_function,l2_score):
        data = {
            "orig": self.params["x"],
            "success": success,
            "adversary": x_adv,
            "l2":l2_score,
            # "loc": loc,
            # "patch": patch,
            #"process": self.process
        }
        if self.i  != None : 
            np.save(os.path.join(self.params["save_directory"],str(self.i )) + '.npy', data, allow_pickle=True)
        else : 
            np.save(os.path.join(self.params["save_directory"], 'ex') +'.npy', data, allow_pickle=True)

    def optimise(self, loss_function):
        # get params
        x = self.params["x"]
        c, h, w = self.params["c"], self.params["h"], self.params["w"]
        s = self.params["s"] 

        # initialize
        patch_geno = np.random.rand(self.params["N"], 7)
        patch = render(patch_geno, s)
        loc = np.random.randint(h - s, size=2)

        update_loc_period = self.params["update_loc_period"]

        x_adv = x.copy()
        x_adv[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch
        x_adv = np.clip(x_adv, 0., 1.)
        success, loss = loss_function(x_adv)
        l2_curr = l2(adv_patch=patch, orig_patch=x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy())

        patch_counter = 0
        n_queries = self.params["n_queries"]
        for it in tqdm(range(1, n_queries)):
            patch_counter += 1
            if patch_counter < update_loc_period:
                patch_new_geno = mutate(patch_geno, self.params["mut"])
                patch_new = render(patch_new_geno, s)
                x_adv_new = x.copy()
                x_adv_new[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch_new
                x_adv_new = np.clip(x_adv_new, 0., 1.)

                # evaluate new sol 
                success_new, loss_new = loss_function(x_adv_new)

                orig_patch = x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy()
                l2_new = l2(adv_patch=patch_new, orig_patch=orig_patch)

                if success == True and success_new == True:

                    if l2_new < l2_curr:
                        loss = loss_new
                        success = success_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new
                else : 
                    if loss_new < loss: 
                        loss = loss_new
                        success = success_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

            else:
                patch_counter = 0

                # location update
                sh_i = int(max(sh_selection(n_queries, it) * h, 0))
                loc_new = loc.copy()
                loc_new = update_location(loc_new, sh_i, h, s)
                x_adv_new = x.copy()
                x_adv_new[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :] = patch
                x_adv_new = np.clip(x_adv_new, 0., 1.)
                # evaluate new solution

                success_new, loss_new = loss_function(x_adv_new)

                orig_patch_new = x[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :].copy()
                l2_new = l2(adv_patch=patch, orig_patch=orig_patch_new)

                if success == True and success_new == True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        success = success_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new
                else : 
                    if loss_new < loss:
                        loss = loss_new
                        success = success_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new
                    elif self.params["temp"] != None: 
                        diff = loss_new - loss
                        curr_temp = self.params["temp"] / (it +1)
                        metropolis = math.exp(-diff/curr_temp)
                        if np.random.rand() < metropolis: 
                            loss = loss_new
                            success = success_new
                            loc = loc_new
                            x_adv = x_adv_new
                            l2_curr = l2_new      
            self.process.append([loc, patch_geno])

        l2_score = l2(self.params["x"],x_adv)
        self.completion_procedure(success, x_adv,n_queries, loc, patch, loss_function,l2_score)
        return x_adv


class Attack_realW(Attack_idealW):
    def optimise(self, loss_function):
        # get params
        x = self.params["x"]
        c, h, w = self.params["c"], self.params["h"], self.params["w"]
        s = self.params["s"] 

        # initialize
        patch_geno = np.random.rand(self.params["N"], 7)
        patch = render(patch_geno, s)
        loc = np.random.randint(h - s, size=2)

        update_loc_period = self.params["update_loc_period"]

        x_adv = x.copy()
        x_adv[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch
        x_adv = np.clip(x_adv, 0., 1.)
        success, loss = loss_function(x_adv)
        l2_curr = l2(adv_patch=patch, orig_patch=x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy())

        patch_counter = 0
        n_queries = self.params["n_queries"]
        for it in tqdm(range(1, n_queries)):
            patch_counter += 1
            if patch_counter < update_loc_period:
                patch_new_geno = mutate(patch_geno, self.params["mut"])
                patch_new = render(patch_new_geno, s)
                x_adv_new = x.copy()
                x_adv_new[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch_new
                x_adv_new = np.clip(x_adv_new, 0., 1.)

                # evaluate new sol 
                success_new, loss_new = loss_function(x_adv_new)

                orig_patch = x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy()
                l2_new = l2(adv_patch=patch_new, orig_patch=orig_patch)

                if success == True and success_new == True:

                    if l2_new < l2_curr:
                        loss = loss_new
                        success = success_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

                elif success_new == True:
                    loss = loss_new
                    success = success_new
                    patch = patch_new
                    patch_geno = patch_new_geno
                    x_adv = x_adv_new
                    l2_curr = l2_new
                elif success ==False and success_new == False : 
                    if loss_new < loss: # minimization
                        loss = loss_new
                        success = success_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

            else:
                patch_counter = 0

                # location update
                sh_i = int(max(sh_selection(n_queries, it) * h, 0))
                loc_new = loc.copy()
                loc_new = update_location(loc_new, sh_i, h, s)
                x_adv_new = x.copy()
                x_adv_new[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :] = patch
                x_adv_new = np.clip(x_adv_new, 0., 1.)
                # evaluate new solution

                success_new, loss_new = loss_function(x_adv_new)

                orig_patch_new = x[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :].copy()
                l2_new = l2(adv_patch=patch, orig_patch=orig_patch_new)

                if success == True and success_new == True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        success = success_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new

                elif success_new == True:
                    loss = loss_new
                    success = success_new
                    loc = loc_new
                    x_adv = x_adv_new
                    l2_curr = l2_new
                elif success ==False and success_new == False :  
                    if loss_new < loss:
                        loss = loss_new
                        success = success_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new
                    elif self.params["temp"] != None: 
                        diff = loss_new - loss
                        curr_temp = self.params["temp"] / (it +1)
                        metropolis = math.exp(-diff/curr_temp)
                        if np.random.rand() < metropolis: 
                            loss = loss_new
                            success = success_new
                            loc = loc_new
                            x_adv = x_adv_new
                            l2_curr = l2_new 

            self.process.append([loc, patch_geno])

         # save result   
        l2_score = l2(self.params["x"],x_adv)
        self.completion_procedure(success, x_adv, n_queries, loc, patch, loss_function,l2_score)
        return x_adv