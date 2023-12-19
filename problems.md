# Basic problem

I used a fine-tuned VGG16 in this kaggle competition. The test result is good (98%). However, when I started to predict and submit, a real tricky problem appears. The problem is that when this for loop:

```python
# prediction

    model.eval()

    probabilities = []

    with torch.no_grad():

        for i, (batch, _) in enumerate(loader):

            batch = batch.cuda() if torch.cuda.is_available() else batch

            outputs = model(batch)

            probabilities.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())

  

    # save results to CSV file

    with open('submissions.csv', 'w', newline='') as csvfile:

        fieldnames = ['id', 'label']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  

        writer.writeheader()

        for i, prob in enumerate(probabilities):

            writer.writerow({'id': i + 1, 'label': prob})
```

Notice that I didn't use the second value which testloader returns (BTW, this return value is determined by the get_item() function of the dataset. When training and testing, this second value is **labels**). Above is the original code i use, which won last place in the competition and it's fair because it's wrong. The id column I intuively use variable *i* in enumerate. But when I use that, I assume the way test_loader load my images in the testdata_dir is as same as I thought, which is by the number of the image. (1.jpg, 2.jpg and so on). And that's where the problem lies. When I finally add a print(label) in the get_item, I found that the way it go through my dir is like this:

```shell
9098
9099
91
910
9100
9101
9102
9103
9104
9105
9106
9107
9108
9109
911
9110
```

# Cause deteced

So it's using lexicographical order to load the images! Yep, simple as it is, I spent 3 hours trying to find what's wrong about my brain, my PC, and the whole fucking world.
Finally, use the number from the file and zip the id with probabilities. Last sort by the id's number then we're all good. Here's the correct code:
```python
    probabilities = []

    with torch.no_grad():

        for i, (batch, fileid) in enumerate(loader):

            # print(image_names)

            batch = batch.cuda() if torch.cuda.is_available() else batch

            outputs = model(batch)

            preds_list = torch.nn.functional.softmax(outputs, dim=1)[:, 1].tolist()

            probabilities += list(zip(list(fileid), preds_list))

    # probabilities.sort(key = lambda x : int(x[0]))

    # save results to csv

    probabilities.sort(key=lambda x: int(x[0]))  # Sort by fileid


    with open('submissions.csv', 'w', newline='') as csvfile:

        fieldnames = ['id', 'label']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

  

        writer.writeheader()

        for i, (fileid, label) in enumerate(probabilities):

            #writer.writerow({'id': i + 1, 'label': label})

            writer.writerow({'id': fileid, 'label': label})
```


# Conclusion

Hope after reading this, you can avoid being this pathetic like me. :)