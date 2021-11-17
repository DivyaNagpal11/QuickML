$(document).on('click', '#visual', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_visual/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#contentdata').html(response['I1']);
            hideloading();
        }
    });
});


$(document).on('click', '#model', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_model/", data).then(function(response){
        if (response !== null || response !== undefined){
           $('#contentdata').html(response['I2']);
           //right_side_bar();
            hideloading();
        }
    });
});


$(document).on('click', '#convolution', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_convolution/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            let isExpanded = $(convolution).attr("aria-expanded");
            let conv_data = response["params"];
            if (conv_data !== 0 & isExpanded === 'true')
            {
                $("#show_conv_table_data tr").remove();
                conv_data.forEach(function (each)
                {
                    let newserialno = document.getElementById("show_conv_table_data").rows.length;
                    if (newserialno === undefined)
                    {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_conv_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4);
                    let col6 = row.insertCell(5);
                    let col7 = row.insertCell(6);
                    let col8 = row.insertCell(7);
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['filter'];
                    col3.innerText = each['kernel_size'];
                    col4.innerText = each["strides"];
                    col5.innerText = each["padding"];
                    col6.innerText = each["activation"];
                    col7.innerHTML = '<button class="btn" id="delete_conv_but">&times;</button>';
                    col8.innerHTML = each['index_no'];
                    col8.style.visibility = 'hidden';
                });
            }
        $("convolution").attr("aria-expanded","false");
        hideloading();
        }
    });
});


$(document).on('click', '#delete_conv_but', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("show_conv_table_data").getElementsByTagName("tr")[row_index];
    let layer_no = row.cells[7].innerHTML;
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(), 'layer_no':layer_no};
    ajax_request("POST", "/nn_model_del_conv/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            if (response["a"] == 0) {
               document.getElementById("show_conv_table_data").deleteRow(row_index);
               hideloading();
            }
            else{
                document.getElementById("alertdata").innerHTML = "<p>Sorry the first layer can't be deleted. </p>";
                $("#alert").addClass("showdiv");
                hideloading();
            }

        }
    });
});


$(document).on('click', '#conv_button', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "filter": $("#filters_id").val(),"kernel": $("#kernel_id").val(), "stride": $("#stride_id").val(),
        "padding": $("#padding_id").val(), "activation": $("#activation_id").val()};
    ajax_request("POST", "/nn_model_conv/", data).then(function(response){
        if (response !== null || response !== undefined){
            $("#show_conv_table_data tr").remove();
            let conv_data = response["params"];
            if (conv_data !== 0) {
                conv_data.forEach(function (each) {
                    let newserialno = document.getElementById("show_conv_table_data").rows.length;
                    if (newserialno === undefined) {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_conv_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4);
                    let col6 = row.insertCell(5);
                    let col7 = row.insertCell(6);
                    let col8 = row.insertCell(7)
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['filter'];
                    col3.innerText = each['kernel_size'];
                    col4.innerText = each["strides"];
                    col5.innerText = each["padding"];
                    col6.innerText = each["activation"];
                    col7.innerHTML = '<button class="btn" id="delete_conv_but">&times;</button>';
                    col8.innerHTML = each['index_no'];
                    col8.style.visibility = 'hidden';
                });
            }
            $('#filters_conv').html(response['I3']);
            hideloading();
        }
    });
});


$(document).on('click', '#MaxPooling', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_maxpool/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            let isExpanded = $(MaxPooling).attr("aria-expanded");
            let maxpool_data = response["params"];
            if (maxpool_data !== 0 & isExpanded === 'true')
            {
                $("#show_maxpool_table_data tr").remove();
                maxpool_data.forEach(function (each)
                {
                    let newserialno = document.getElementById("show_maxpool_table_data").rows.length;
                    if (newserialno === undefined)
                    {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_maxpool_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4);
                    let col6 = row.insertCell(5)
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['pool_size'];
                    col3.innerText = each['strides'];
                    col4.innerText = each["padding"];
                    col5.innerHTML = '<button class="btn" id="delete_max_but">&times;</button>';
                    col6.innerHTML = each['index_no'];
                    col6.style.visibility = 'hidden';
                });
            }
        $("MaxPooling").attr("aria-expanded","false");
        hideloading();
        }
    });
});


$(document).on('click', '#delete_max_but', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("show_maxpool_table_data").getElementsByTagName("tr")[row_index];
    let layer_no = row.cells[5].innerHTML;
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(), 'layer_no':layer_no};
    ajax_request("POST", "/nn_model_del_maxpool/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            document.getElementById("show_maxpool_table_data").deleteRow(row_index);
            hideloading();
        }
    });
});


$(document).on('click', '#max_pool_button', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "pool_id": $("#pool_id").val(), "strides": $("#stride").val(), "padding": $("#padding").val()};
    ajax_request("POST", "/nn_model_max_pool/", data).then(function(response){
        if (response !== null || response !== undefined){
             $("#show_maxpool_table_data tr").remove();
            let maxpool_data = response["params"];
            if (maxpool_data !== 0) {
                maxpool_data.forEach(function (each) {
                    let newserialno = document.getElementById("show_maxpool_table_data").rows.length;
                    if (newserialno === undefined) {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_maxpool_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4);
                    let col6 = row.insertCell(5)
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['pool_size'];
                    col3.innerText = each['strides'];
                    col4.innerText = each["padding"];
                    col5.innerHTML = '<button class="btn" id="delete_max_but">&times;</button>';
                    col6.innerHTML = each['index_no']
                    col6.style.visibility = 'hidden';
                });
            }
             $('#filters_max').html(response['I4']);
            hideloading();
        }
    });
});


$(document).on('click', '#DropOut', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_dropout/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            let isExpanded = $(DropOut).attr("aria-expanded");
            let dropout_data = response["params"];
            if (dropout_data !== 0 & isExpanded === 'true')
            {
                $("#show_dropout_table_data tr").remove();
                dropout_data.forEach(function (each)
                {
                    let newserialno = document.getElementById("show_dropout_table_data").rows.length;
                    if (newserialno === undefined)
                    {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_dropout_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4)
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['rate'];
                    col3.innerText = each['seed'];
                    col4.innerHTML = '<button class="btn" id="delete_drop_but">&times;</button>';
                    col5.innerHTML = each['index_no'];
                    col5.style.visibility = 'hidden';
                });
            }
        $("DropOut").attr("aria-expanded","false");
        hideloading();
        }
    });
});


$(document).on('click', '#delete_drop_but', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("show_dropout_table_data").getElementsByTagName("tr")[row_index];
    let layer_no = row.cells[4].innerHTML;
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(), 'layer_no':layer_no};
    ajax_request("POST", "/nn_model_del_dropout/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            document.getElementById("show_dropout_table_data").deleteRow(row_index);
            hideloading();
        }
    });
});


$(document).on('click', '#drop_button', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "rate": $("#rate_id").val(), "seed": $("#seed").val()};
    ajax_request("POST", "/nn_model_drop/", data).then(function(response){
        if (response !== null || response !== undefined){
            let isExpanded = $(DropOut).attr("aria-expanded");
            let dropout_data = response["params"];
            if (dropout_data !== 0 & isExpanded === 'true')
            {
                $("#show_dropout_table_data tr").remove();
                dropout_data.forEach(function (each)
                {
                    let newserialno = document.getElementById("show_dropout_table_data").rows.length;
                    if (newserialno === undefined)
                    {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_dropout_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4)
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['rate'];
                    col3.innerText = each['seed'];
                    col4.innerHTML = '<button class="btn" id="delete_drop_but">&times;</button>';
                    col5.innerHTML = each['index_no'];
                    col5.style.visibility = 'hidden';
                });
            }
              $('#filters_drop').html(response['I5']);
            hideloading();
        }
    });
});


$(document).on('click', '#flatten_button', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_model_flatten/", data).then(function(response){
        if (response !== null || response !== undefined)
        {
            hideloading();
        }
    });
});


$(document).on('click', '#Dense', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_dense/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            let isExpanded = $(Dense).attr("aria-expanded");
            let dense_data = response["params"];
            if (dense_data !== 0 & isExpanded === 'true')
            {
                $("#show_dense_table_data tr").remove();
                dense_data.forEach(function (each)
                {
                    let newserialno = document.getElementById("show_dense_table_data").rows.length;
                    if (newserialno === undefined)
                    {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_dense_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4)
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['units'];
                    col3.innerText = each['activation'];
                    col4.innerHTML = '<button class="btn" id="delete_dense_but">&times;</button>';
                    col5.innerHTML = each['index_no'];
                    col5.style.visibility = 'hidden';
                });
            }
        $("Dense").attr("aria-expanded","false");
        hideloading();
        }
    });
});


$(document).on('click', '#delete_dense_but', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let row_index = $(this).closest('tr').index();
    let row = document.getElementById("show_dense_table_data").getElementsByTagName("tr")[row_index];
    let layer_no = row.cells[4].innerHTML;
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(), 'layer_no':layer_no};
    ajax_request("POST", "/nn_model_del_dense/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            document.getElementById("show_dense_table_data").deleteRow(row_index);
            hideloading();
        }
    });
});


$(document).on('click', '#dense_button', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "unit": $("#unit_id").val(), "activation": $("#activation").val()};
    ajax_request("POST", "/nn_model_dense/", data).then(function(response){
        if (response !== null || response !== undefined){
            let isExpanded = $(Dense).attr("aria-expanded");
            let dense_data = response["params"];
            if (dense_data !== 0 & isExpanded === 'true')
            {
                $("#show_dense_table_data tr").remove();
                dense_data.forEach(function (each)
                {
                    let newserialno = document.getElementById("show_dense_table_data").rows.length;
                    if (newserialno === undefined)
                    {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_dense_table_data");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    let col5 = row.insertCell(4)
                    col1.innerText = newserialno + 1;
                    col2.innerText = each['units'];
                    col3.innerText = each['activation'];
                    col4.innerHTML = '<button class="btn" id="delete_dense_but">&times;</button>';
                    col5.innerHTML = each['index_no'];
                    col5.style.visibility = 'hidden';
                });
            }
            hideloading();
        }
    });
});


$(document).on('click', '#Fitting', function (event)
{
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_fit/", data).then(function(response)
    {
        if (response !== null || response !== undefined)
        {
            let isExpanded = $(Fitting).attr("aria-expanded");
            let fit_data = response["params"];
            if (fit_data !== 0 & isExpanded === 'true')
            {
                $("#show_fit tr").remove();
                fit_data.forEach(function (each)
                {
                    let newserialno = document.getElementById("show_fit").rows.length;
                    if (newserialno === undefined)
                    {
                        newserialno = 0;
                    }
                    let table = document.getElementById("show_fit");
                    let row = table.insertRow(newserialno);
                    let col1 = row.insertCell(0);
                    let col2 = row.insertCell(1);
                    let col3 = row.insertCell(2);
                    let col4 = row.insertCell(3);
                    col1.innerText = each['batch_size'];
                    col2.innerText = each['epochs'];
                    col3.innerText = each['verbose'];
                    col4.innerText = each['val_split'];
                });
            }
        $("Fitting").attr("aria-expanded","false");
        hideloading();
        }
    });
});


$(document).on('click', '#fit_button', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "batch_id": $("#batch_id").val(), "epochs": $("#epochs").val(), "verbose": $("#verbose").val(),
        "val": $("#val").val()};
    ajax_request("POST", "/nn_model_fit/", data).then(function(response){
        if (response !== null || response !== undefined){
            if (response["a"] == 0) {
                right_side_bar();
                hideloading();
            }
            else{
                document.getElementById("alertdata").innerHTML = "<p>Please add the last layer's output units same as the no of categories </p>";
                $("#alert").addClass("showdiv");
                hideloading();
            }
        }
    });
});


$(document).on('click', '#validate', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let data = {"project_id": project_id, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/nn_validate/", data).then(function(response){
        if (response !== null || response !== undefined){
           $('#contentdata').html(response['I6']);
            hideloading();
        }
    });
});


function PreviewImage() {
    let oFReader = new FileReader();
    oFReader.readAsDataURL(document.getElementById("uploadImage").files[0]);
    oFReader.onload = function (oFREvent) {
        document.getElementById("uploadPreview").src = oFREvent.target.result;
    };
}


$(document).on('click','#submit_button',function(e) {
    e.preventDefault();
    showloading();
    let data = new FormData();
    let img = $('#uploadImage')[0].files[0];
    data.append('img', img);
    data.append('csrfmiddlewaretoken',$('input[name=csrfmiddlewaretoken]').val());
    data.append('project_id',$("#selectprojectbtn").val());
    ajax_request("POST", "/nn_validate_image/", data).then(function(response){
        if (response !== null || response !== undefined){
             document.getElementById("predicted_output").innerText = response['predict'];
            hideloading();
        }
    });
});


