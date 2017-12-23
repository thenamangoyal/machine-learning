re_error = 0.1;
[rawX, label] = readdata();
[rawprojX, U, ~] = runpca(rawX, re_error);

fprintf('Required components %d\n', size(U,2));
% Display Component
figure;
comp_to_disp=[1 2 3];

np = size(comp_to_disp,2);
for sp=1:np
    subplot(1,np,sp);
    imshow(reshape(U(:,comp_to_disp(sp)),20,20));
    title(['Principal Component ',num2str(comp_to_disp(sp))]);
end
% Display Reconstruction
figure;
% title(['Recosntruction Error ',num2str(re_error) ]);
points_to_disp=[1 4000 1000];
np = size(points_to_disp,2);
for sp=1:np
    subplot(np,2,2*sp-1);
    imshow(reshape(rawX(points_to_disp(sp),:),20,20));

    title(['Original Image ',num2str(points_to_disp(sp))]);
    subplot(np,2,2*sp);
    imshow(reshape(rawprojX(points_to_disp(sp),:),20,20));
    title(['Reconstructed Image ',num2str(points_to_disp(sp))]);
end