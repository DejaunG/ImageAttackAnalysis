$(document).ready(function() {
    // Navigation functionality
    $('#generate-btn').click(function(e) {
        e.preventDefault();
        $(this).addClass('active');
        $('#detect-btn').removeClass('active');
        $('#generate-section').fadeIn();
        $('#detect-section').hide();
    });

    $('#detect-btn').click(function(e) {
        e.preventDefault();
        $(this).addClass('active');
        $('#generate-btn').removeClass('active');
        $('#detect-section').fadeIn();
        $('#generate-section').hide();
    });

    // Generate adversarial image functionality
    $('#myFile').on('change', function() {
        handleFileUpload(this, '/upload', '#uploaded-image', '#adversarial-examples', '#training-history');
    });

    // Detect and revert functionality
    $('#detectFile').on('change', function() {
        handleFileUpload(this, '/detect', '#detect-uploaded-image', '#detect-results');
    });

    function handleFileUpload(fileInput, endpoint, imageSelector, resultsSelector, historySelector = null) {
        var formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Show loading modal
        $('#loadingModal').modal('show');

        $.ajax({
            url: endpoint,
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
    console.log('Server response:', data);
    $(imageSelector).attr('src', '/uploads/' + data.filename).hide().fadeIn();

    // Clear previous results
    $(resultsSelector).empty();

    // Display results
    data.images.forEach(function(image, index) {
        if (image.includes('detected_reverted.png')) {
            $(resultsSelector).append(`
                <img src="/${image}" alt="Detected and Reverted Image" class="img-fluid mb-3">
            `).hide().fadeIn();
        }
    });

    // Display adversarial detection result
    if (data.adversarial_result) {
        $(resultsSelector).append(`
            <div class="adversarial-result">
                <h3>Adversarial Detection Result</h3>
                <pre>${data.adversarial_result}</pre>
            </div>
        `).hide().fadeIn();
    }

    console.log('Success:', data);
},
            error: function(xhr, status, error) {
                $('#response').html(`<strong>Error:</strong> ${xhr.responseText}`).fadeIn().delay(3000).fadeOut();
                console.log('Error:', error);
            },
            complete: function() {
                // Hide loading modal
                $('#loadingModal').modal('hide');
            }
        });
    }
});