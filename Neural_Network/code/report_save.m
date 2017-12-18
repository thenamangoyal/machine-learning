function report_save(Trainerror, Valerror,trainN, valN, eta, p,m,beginame,matpath, figpath, pngpath)
nEpochs = size(Trainerror,1);
mseTrainerror = Trainerror/trainN;
mseValerror = Valerror/valN;

commontitle= ['Eta:', num2str(eta),' minibatch:',num2str(m), ' Dropout:', num2str(p)];
%Matrix Save
name = strcat('_epochs',num2str(nEpochs),'_eta',num2str(eta),'_p',num2str(p),'_m',num2str(m));
% save(strcat(matpath,filesep,beginame,'Sum_Train',name,'.mat'),'Trainerror');
% save(strcat(matpath,filesep,beginame,'Sum_Val',name,'.mat'),'Valerror');
save(strcat(matpath,filesep,beginame,'MSE_Train',name,'.mat'),'mseTrainerror');
save(strcat(matpath,filesep,beginame,'MSE_Val',name,'.mat'),'mseValerror');

% % Sum Train
% figure; plot(1:nEpochs, Trainerror, 'b', 'LineWidth', 2);,
% title({'Sum of squares Training error',commontitle});
% 
% xlabel('no of Epochs');
% ylabel('Sum of squares Training error');
% savefig(strcat(figpath,filesep,beginame,'Sum_Train',name,'.fig'));
% saveas(gcf, strcat(pngpath,filesep,beginame,'Sum_Train',name,'.png'));
% 
% % MSE Train
% figure; plot(1:nEpochs, mseTrainerror, 'b', 'LineWidth', 2);
% title({'MSE of squares Training error',commontitle});
% 
% xlabel('no of Epochs');
% ylabel('MSE of squares Training error');
% savefig(strcat(figpath,filesep,beginame,'MSE_Train',name,'.fig'));
% saveas(gcf, strcat(pngpath,filesep,beginame,'MSE_Train',name,'.png'));
% 
% % Sum Val
% figure; plot(1:nEpochs, Valerror, 'r', 'LineWidth', 2);
% title({'Sum of squares Validation error',commontitle});
% 
% xlabel('no of Epochs');
% ylabel('Sum of squares Validation error');
% savefig(strcat(figpath,filesep,beginame,'Sum_Val',name,'.fig'));
% saveas(gcf, strcat(pngpath,filesep,beginame,'Sum_Val',name,'.png'));
% 
% % MSE Val
% figure; plot(1:nEpochs, mseValerror, 'r', 'LineWidth', 2);
% title({'MSE of squares Validation error',commontitle});
% 
% xlabel('no of Epochs');
% ylabel('MSE of squares Validation error');
% savefig(strcat(figpath,filesep,beginame,'MSE_Val',name,'.fig'));
% saveas(gcf, strcat(pngpath,filesep,beginame,'MSE_Val',name,'.png'));
% 
% Sum Train and Sum Val
figure; plot(1:nEpochs, Trainerror, 'b', 'LineWidth', 2); hold on;
plot(1:nEpochs, Valerror, 'r', 'LineWidth', 2);
legend('Train', 'Validation', 'Location', 'SouthOutside', ...
    'Orientation', 'horizontal');
title({'Sum of squares Training and Validation error',commontitle});

xlabel('no of Epochs');
ylabel('Sum of squares Training and Validation error');
savefig(strcat(figpath,filesep,beginame,'Sum_Train&Val',name,'.fig'));
saveas(gcf, strcat(pngpath,filesep,beginame,'Sum_Train&Val',name,'.png'));

% MSE Train and Sum Val
figure; plot(1:nEpochs, mseTrainerror, 'b', 'LineWidth', 2); hold on;
plot(1:nEpochs, mseValerror, 'r', 'LineWidth', 2);
legend('Train', 'Validation', 'Location', 'SouthOutside', ...
    'Orientation', 'horizontal');
title({'MSE Training and Validation error',commontitle});

xlabel('no of Epochs');
ylabel('MSE Training and Validation error');
savefig(strcat(figpath,filesep,beginame,'MSE_Train&Val',name,'.fig'));
saveas(gcf, strcat(pngpath,filesep,beginame,'MSE_Train&Val',name,'.png'));

end
