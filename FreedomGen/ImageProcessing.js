$(document).ready(function() {
    $('#upload-form').on('change', '#myFile', function() {
        var formData = new FormData($('#upload-form')[0]);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            // In the success function of the AJAX call
            success: function(data) {
                console.log('Server response:', data);
                $('#uploaded-image').attr('src', '/uploads/' + data.filename);
                console.log('Success:', data);
                $('#results').html('Generated Images: ' + data.script_output);

                // Clear the results div
                $('#results').empty();

                // Display the images
                data.images.forEach(function(image) {
                    $('#results').append('<div class="image-container"><img src="/' + image + '"></div>');
                });
            },
            error: function(xhr, status, error) {
                $('#response').html('Error: ' + xhr.responseText);
                console.log('Error:', error);
            }
        });
    });
});