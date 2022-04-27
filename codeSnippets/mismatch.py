def mismatch(encoder, model, data_loader, criterion, optimizer):
    model.eval()    
    encoder.eval()
    model.register_hook=True
    encoder.register_hook=True
    model.reset_hook()
    encoder.reset_hook()
    mm_image = None
    losses = []
    for idx, (image1, image2, labels) in enumerate(data_loader): 
        image1, image2, labels = image1.to(device), image2.to(device), (labels.float()).to(device)
        # halt the visual flow
        if idx in range(400,500):
            if mm_image is None:
                mm_image = image1
            image1 = mm_image
            image2 = mm_image
        encoded1 = encoder(image1)
        encoded2 = encoder(image2)

        batch_size1 = len(encoded1)
        batch_size2 = len(encoded2)

        output1 = encoded1.reshape(batch_size1,1,-1)
        output2 = encoded2.reshape(batch_size2,1,-1)
        
        seq = torch.cat((output1, output2.detach()), dim=1)

        outputs = model(seq.to(device))
        
        optimizer.zero_grad()
        loss = criterion(outputs[:,-1].squeeze(), labels.squeeze())
        losses.append(loss.item())
        loss.backward()