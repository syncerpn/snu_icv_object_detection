function rect_box = ellipse_to_rect_box(ellipse_box)
% Convert ellipse annotations to rectangular ones

sin2 = (sin(ellipse_box(:,3))).^2;
cos2 = (cos(ellipse_box(:,3))).^2;
a2 = (ellipse_box(:,1)).^2;
b2 = (ellipse_box(:,2)).^2;

h = sqrt(1./(sin2./a2 + cos2./b2))*2;
w = sqrt(1./(sin2./b2 + cos2./a2))*2;

rect_box = [max(ellipse_box(:,4)-w/2,1) max(ellipse_box(:,5)-h/2,1) ellipse_box(:,4)+w/2 ellipse_box(:,5)+h/2];
rect_box = round(rect_box);