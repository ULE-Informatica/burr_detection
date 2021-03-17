import cv2
import os
import csv
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from skimage.morphology import disk


def potentialImage(img,width,from_section,to_section,number_sections):
    dim_width = int(img.shape[0]/width)
    dim_heigth = int(img.shape[1])
    from_width = dim_width*(max(from_section-number_sections,0))
    to_width = dim_width*(min(to_section+number_sections,width-1))
    #print(from_width, to_width)
    #cv2.rectangle(img,(0,from_width),(dim_heigth,to_width),(0,255,0),3)
    return img[from_width:to_width+dim_width,:]

class Line:
    """
    def __init__(self, x1, y1, x2, y2):
        self.m = (y2-y1)/(x2-x1)
        self.n = -((x1*(y2-y1))/(x2-x1))+y1
        self.A = (y2-y1)
        self.B = -(x2-x1)
        self.C = -x1*(y2-y1)+y1*(x2-x1)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    """

    def __init__(self, m, n):
        if (m==0):
            m=1
        self.m = m
        self.n = n
        self.y1 = 0
        self.x1 = int((self.y1 - self.n) /self.m)
        self.y2 = 100
        self.x2 = int((self.y2 - self.n) /self.m)

    def distance(self, x0, y0):
        return abs((self.A*x0+self.B*y0+self.C)/(math.sqrt(self.A**2+self.B**2)))

def sections(img, width=1, heigth=1):
    sections_list = []
    dim_width = int(img.shape[0]/width)
    dim_heigth = int(img.shape[1]/heigth)
    total_pixels_section = dim_width*dim_heigth
    for i in range(0,width):
        for j in range(0,heigth):
            pos_width = dim_width*i
            pos_heigth = dim_heigth*j
            img_cut=img[pos_width:pos_width+dim_width,pos_heigth:pos_heigth+dim_heigth]
            sections_list.append((np.count_nonzero(img_cut == 255)*100)/total_pixels_section)
    return sections_list

def potentialSections(sections_list):
    selection = []
    for i in range(0,len(section_list)-1):
        diff = abs(section_list[i] - section_list[i+1])
        if(diff>5):
            if(i not in selection):
                selection.append(i)
            selection.append(i+1)

    groups = []
    groups_diff = []
    for i in range(0, len(selection)-1):
        diff = abs(selection[i] - selection[i+1])
        if len(groups)==0 or (diff>5):
            group = []
            groups.append(group)
            group_diff = []
            groups_diff.append(group_diff)
        group = groups[-1]
        if (diff<5) and selection[i] not in group:
            group.append(selection[i])
        group.append(selection[i+1])
        diff_v = abs(section_list[selection[i]] - section_list[selection[i+1]])
        group_diff = groups_diff[-1]
        if (diff<5):
            group_diff.append(diff_v)

    max_value=0
    max_group = 0
    for i in range(0,len(groups_diff)):
        max_v = max(groups_diff[i])
        if max_v>max_value:
            max_value = max_v
            max_group = i
    if len(groups)>0:
        potentialGroup = groups[max_group]
    else:
        potentialGroup = []
    return [selection, potentialGroup]

def polynomial(X,y,degree):
    try:
        reg = linear_model.LinearRegression().fit(X, y)
        line = Line(reg.coef_[0], reg.intercept_)
    except ValueError:
        if y[0] == y[-1]:
            line = Line(1, 0)
        else:
            m = (y[-1]-y[0])/(X[-1]-X[0])
            n = -((X[0]*(y[-1]-y[0]))/(X[-1]-X[0]))+y[0]
            line = Line(m, n)
    return line

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rustico", required=True, help="RUSTICO Images folder path")
#ap.add_argument("-i", "--images", required=True, help="Images folder path")
ap.add_argument("-o", "--original", required=True, help="Orginal Images folder path")
ap.add_argument("-f", "--folder", required=True, help="Folder to save results images")

args = vars(ap.parse_args())

sections_width = 90
sections_num = 35
sections_number = 100

if not os.path.exists(args['folder']):
    os.makedirs(args['folder'])

csv_file='info_h_'+str(sections_width)+'.csv'#+'_'+str(sections_num)+'.csv'
data_file = open(csv_file, "w", newline="",encoding="utf-8")
data_file.close()
headers = ['img_name','folder','image']  + ['h_'+str(i) for i in range(1,sections_width+1)] + ['pw_'+str(i) for i in range(1,sections_width+1)]
with open(csv_file, mode='a+', newline="",encoding="utf-8") as data_file:
	data_writer = csv.writer(data_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	data_writer.writerow(headers)

for image in os.listdir(args['rustico']):
    if image.endswith('.png'):
        dir = os.path.join(args['rustico'], image)
        #image = "Pieza_12_foto0540.png"
        img_name = image[:-4]
        folder_pieza = img_name[:8]
        img_pieza = img_name[9:]
        print(img_name)

        img = cv2.imread(os.path.join(args['rustico'], image), cv2.IMREAD_GRAYSCALE)
        original_img =  np.array(Image.open(os.path.join(args['original'], folder_pieza, img_pieza+'.tif')))
        #Morphology
        kernel = disk(67)
        #kernel = np.ones((17,77),np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        #Percentage of white pixels and potential section
        section_list = sections(closing,sections_number)
        selection, potential_sections = potentialSections(section_list)
        if len(potential_sections)>0:
            section_list_np = np.asarray(section_list).astype(np.float)
            potential_section = section_list_np[selection]
            potential_values = section_list_np[potential_sections]
            #Potential image
            img_potential = potentialImage(original_img,sections_number,potential_sections[0],potential_sections[-1],3)
            img_potential_RUSTICO = potentialImage(img,sections_number,potential_sections[0],potential_sections[-1],3)
            cv2.imwrite(os.path.join(args['folder'], img_name+'_potential.jpg'),cv2.cvtColor(img_potential, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(args['folder'], img_name+'_potential.png'),img_potential_RUSTICO)
            #Plot process
            f, axs = plt.subplots(1,5,figsize=(20,3))
            axs[0].imshow(original_img)
            axs[0].axis('off')
            axs[0].title.set_text('Original')
            axs[1].imshow(img, cmap='gray')
            axs[1].axis('off')
            axs[1].title.set_text('RUSTICO')
            axs[2].imshow(closing, cmap='gray')
            axs[2].axis('off')
            axs[2].title.set_text('Morphological operations')
            axs[3].plot( range(0,sections_number), section_list, '.')
            axs[3].plot( selection, potential_section, '+')
            axs[3].plot( potential_sections, potential_values, '*')
            axs[3].title.set_text('Study of white pixel percetage')
            axs[3].set_ylabel('White pixel percetage')
            axs[4].imshow(img_potential)
            axs[4].axis('off')
            axs[4].title.set_text('Potential image')
            #plt.show()
            #exit()
            plt.savefig(os.path.join(args['folder'], img_name+'_process.jpg'), dpi=200)
            plt.close('all')

            #Morphology (potential image)
            kernel = np.ones((31,79),np.uint8)
            #kernel = disk(17)
            #closing_potential = cv2.morphologyEx(img_potential_RUSTICO, cv2.MORPH_CLOSE, kernel)
            closing_potential = cv2.dilate(img_potential_RUSTICO,kernel,iterations = 1)

            #Add borders and morhpology
            section_list = sections(closing_potential,10)
            widthMask = closing_potential.shape[1]
            heightMask = closing_potential.shape[0]
            if section_list[0]>0.5:
                mask1 = np.ones((10, widthMask), np.uint8)*255
                mask2 = np.zeros((10, widthMask), np.uint8)
            else:
                mask1 = np.zeros((10, widthMask), np.uint8)
                mask2 = np.ones((10, widthMask), np.uint8)*255
            mask3 = np.ones((heightMask + 20, 10), np.uint8)*255

            closing_mask = np.vstack((mask1, closing_potential))
            closing_mask = np.vstack((closing_mask, mask2))
            closing_mask = np.hstack((mask3, closing_mask))
            closing_mask = np.hstack((closing_mask, mask3))
            closing_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(closing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                added_image = cv2.fillPoly(closing_mask, [cnt], (255))
            closing_mask = closing_mask[10:-10,10:-10]

            #Potential image and mask
            image_mask = cv2.bitwise_not(closing_mask)
            img2 = cv2.merge((image_mask,image_mask,image_mask))
            #added_image = img_potential.copy()
            #added_image = cv2.merge((closing_mask,closing_mask,closing_mask))
            added_image = cv2.addWeighted(img2,0.1,img_potential,0.4,0)


            #rectangle
            heigths = []
            sections_list = []
            final = []
            #f, axs = plt.subplots(1,sections_width)
            for i in range(0,sections_width):
                dim_heigth = int(closing_mask.shape[1]/sections_width)
                closing1 = closing_mask[:, i*dim_heigth:(i+1)*dim_heigth]
                contours, hierarchy = cv2.findContours(closing1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                img2 = cv2.merge((closing1,closing1,closing1))
                if(len(contours)>0):
                    cnt = contours[0]
                    area_max = 0
                    for cn in contours:
                        area = cv2.contourArea(cn)
                        if area > area_max:
                            area_max = area
                            cnt = cn
                    hull = cv2.convexHull(cnt)
                    x,y,w,h = cv2.boundingRect(cnt)
                    heigths.append(h)
                    cut1 = closing1[x:x+w, y:y+h]
                    total_pixels_section = w*h
                    sections_list.append((np.count_nonzero(cut1 == 255)*100)/total_pixels_section)
                    img2 = cv2.drawContours(img2, [cnt], -1, (0,255,0), 2)
                    img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
                    if len(final) ==0:
                        final = img2.copy()
                    else:
                        final = np.hstack((final,img2))
                #axs[i].imshow(img2)
                #axs[i].axis('off')
            #plt.show()


            #Plot potential image process
            f, axs = plt.subplots(6,1,figsize=(15,10))
            axs[0].imshow(img_potential)
            axs[0].axis('off')
            axs[0].title.set_text('Original')
            axs[1].imshow(img_potential_RUSTICO, cmap='gray')
            axs[1].axis('off')
            axs[1].title.set_text('RUSTICO')
            axs[2].imshow(closing_potential, cmap='gray')
            axs[2].axis('off')
            axs[2].title.set_text('Morphological operations')
            axs[3].imshow(closing_mask, cmap='gray')
            axs[3].axis('off')
            axs[3].title.set_text('Morphological operations')
            axs[4].imshow(added_image)
            axs[4].axis('off')
            axs[4].title.set_text('Final result')
            axs[5].imshow(final)
            axs[5].axis('off')
            axs[5].title.set_text('Final result')
            #plt.show()
            #exit()
            plt.savefig(os.path.join(args['folder'], img_name+'_potential_process.jpg'), dpi=200)
            plt.close('all')

            """
            list_m = []
            for i in range(0,sections_width):
                dim_heigth = int(closing_mask.shape[1]/sections_width)
                closing1 = closing_mask[:, i*dim_heigth:(i+1)*dim_heigth]
                section_list = sections(closing1,sections_num)
                list_m += section_list
                """
            """
            selection, potential_sections = potentialSections(section_list)
            if len(potential_sections)>1:
                section_list_np = np.asarray(section_list).astype(np.float)
                potential_section = section_list_np[selection]
                potential_values = section_list_np[potential_sections]
                line = polynomial( np.asarray([potential_sections]).transpose(),np.asarray(potential_values).transpose(),1)
                list_m.append(line.m)
            """
            if len(heigths)==sections_width:#*sections_num:
                # ['img_name','folder','image']  + ['m_'+str(i) for i in range(1,sections_num+1)]
                data_final = [img_name, int(folder_pieza[-2:]), int(img_pieza[4:])] + heigths + sections_list
                with open(csv_file, mode='a+', newline="",encoding="utf-8") as data_file:
                    data_writer = csv.writer(data_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    data_writer.writerow(data_final)
