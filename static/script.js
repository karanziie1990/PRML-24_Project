document.getElementById('select-button').addEventListener('click', function() {
    var queryNumber = document.getElementById('query-number-input').value - 1;

    // Send the query number to the backend
    axios.post('/query', { queryNumber: queryNumber })
        .then(function(response) {
            // Handle the response by displaying the query image
            var queryImageUrl = response.data.queryImage;
            var queryImgElement = document.createElement('img');
            queryImgElement.src = queryImageUrl;
            document.getElementById('query-image-container').innerHTML = '';  // Clear previous content
            document.getElementById('query-image-container').appendChild(queryImgElement);
        })
        .catch(function(error) {
            console.error('Error:', error);
        })
})

document.getElementById('retrieve-button').addEventListener('click', function() {
    var queryNumber = document.getElementById('query-number-input').value - 1;

    // Send the query number to the backend
    axios.post('/retrieve', { queryNumber: queryNumber })
        .then(function(response) {
            // Handle the response by displaying the similar images
            var similarImages = response.data.similarImages;

            // Clear previous content
            document.getElementById('similar-images-container').innerHTML = '';

            // Loop through the similar images and create <img> elements for each
            similarImages.forEach(function(imageUrl) {
                var imgElement = document.createElement('img');
                imgElement.src = imageUrl;
                imgElement.classList.add('image-grid-script'); // Add class for styling
                document.getElementById('similar-images-container').appendChild(imgElement);
            });
        })
        .catch(function(error) {
            console.error('Error:', error);
        })
})

