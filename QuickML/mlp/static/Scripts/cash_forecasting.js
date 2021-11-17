$(document).on('click', '#basic_stats', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();
    let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/cf_basicstats/", data).then(function(response){
        if (response !== null || response !== undefined){
            console.log(response['basicStats']);
            if (be_name === "All Adventist W"){
                $('#contentdata').html(response['basic_stats_html']);
                $('#statistics').html(response['basicStats']);
                hideloading();
            }
            else {
                $('#contentdata').html(response['basic_stats_html']);
                $('#statistics').html(response['basicStats']);
                $('#months_count').html(response['months_count']);
                $('#be_type').html(response['category']);
                $('#daily_boxplot').html(response['daily_box_plot']);
                $('#weekly_boxplot').html(response['weekly_box_plot']);
                $('#monthly_boxplot').html(response['monthly_box_plot']);
                hideloading();
            }
        }
    });
});

$(document).on('click', '#time_series', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();
    let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/cf_timeseries/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#contentdata').html(response['ts_html']);
            $('#ts_rolling_plot').html(response['rolling_plot_html']);
            $('#ts_line_chart').html(response['line_chart_html']);
            $('#ts_bar_plot').html(response['bar_plot_html']);
            hideloading();
        }
    });
});

$(document).on('click', '#weekdays', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();
    let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/cf_weekdays/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#contentdata').html(response['wd_html']);
            $('#wd_bar_plot').html(response['wd_bar_plot_html']);
            $('#wd_line_chart').html(response['wd_line_chart_html']);
            //$('#wd_decompose').html(response['wd_decompose_graph_html']);
            hideloading();
        }
    });
});

$(document).on('click', '#encounter_class', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();
    let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/cf_encounter_class/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#contentdata').html(response['ec_html']);
            $('#ec_line_chart').html(response['ec_line_chart_html']);
            $('#ec_bar_plot').html(response['ec_bar_plot_html']);
            $('#ec_footfall').html(response['ec_footfall_html']);
            $('#ec_charges').html(response['ec_charges_html']);
            hideloading();
        }
    });
});

$(document).on('click', '#financial_class', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();
    let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/cf_financial_class/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#contentdata').html(response['fc_html']);
            $('#fc_line_chart').html(response['fc_line_chart_html']);
            $('#fc_bar_plot').html(response['fc_bar_plot_html']);
            $('#fc_footfall').html(response['fc_footfall_html']);
            $('#fc_charges').html(response['fc_charges_html']);
            hideloading();
        }
    });
});

// $(document).on('click', '#xregs', function (event) {
//     showloading();
//     event.preventDefault();
//     let project_id = $("#selectprojectbtn").val();
//     let be_name = $("#select_billing").val();
//     let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
//     ajax_request("POST", "/cf_xregs/", data).then(function(response){
//         if (response !== null || response !== undefined){
//             $('#contentdata').html(response['xr_html']);
//             hideloading();
//         }
//     });
// });
//
//
$(document).on('change', '#select_xreg', function (event) {
    showloading();
    let x_regs = $(this).val();
    console.log(x_regs);
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();

    let data = {"project_id": project_id,"be_name":be_name, "x_regs":x_regs, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    console.log(data);
    ajax_request("GET", "/cf_xregs_select/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#ts_xregs_plots').html(response['ts_ext_reg_plot_html'])
            $('#xregs_plots').html(response['lagged_plot_html']);
            $('#xregs_co-realation_plots').html(response['exreg_correalation_plot_html']);
            $('#xregs_co-realation_stats').html(response['exreg_correalation_stats_html']);
            hideloading();
        }
    });
});


$(document).on('click', '#xregs', function (event) {
    event.preventDefault();
    showloading();
    let project_id = $("#selectprojectbtn").val();
    let data = {
        'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
        "project_id": project_id
     };
    ajax_request("POST", "/cf_xregs/", data).then(function (response) {
            if (response !== null || response !== undefined) {
                $('#contentdata').html(response['xr_html']);
                $('#xregs_plots').html(response['lagged_plot_html']);
                $('#xregs_co-realation_plots').html(response['exreg_correalation_plot_html']);
                $('#xregs_co-realation_stats').html(response['exreg_correalation_stats_html']);
                hideloading();
            }
        });
});


// function exregs_plot() {
//     showloading();
//     let be_name = $("#select_billing").val();
//     let ex_reg = $('#select_xreg').val();
//     let project_id = $("#selectprojectbtn").val();
//     let data = {
//         'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val(),
//         "project_id": project_id,
//         "ex_reg": ex_reg,
//         "be_name":be_name
//     };
//     ajax_request("POST", "/cf_xregs_select/", data).then(function (response) {
//         if (response !== null || response !== undefined) {
//             $('#xregs_plots').html(response['lagged_plot_html']);
//             $('#xregs_co-realation_plots').html(response['exreg_correalation_plot_html']);
//             $('#xregs_co-realation_stats').html(response['exreg_correalation_stats_html']);
//             hideloading();
//         }
//
//     });
// }



$(document).on('click', '#acf_pacf', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();

    let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/cf_acf_pacf/", data).then(function(response){
        if (response !== null || response !== undefined){
            $('#contentdata').html(response['acf_pacf_html'])
            $('#acf_plot').html(response['acf_plot_html']);
            $('#pacf_plot').html(response['pacf_plot_html']);
            hideloading();
        }
    });
});


// $(document).on('click', '#forecast', function (event) {
//     showloading();
//     event.preventDefault();
//     let project_id = $("#selectprojectbtn").val();
//     let be_name = $("#select_billing").val();
//     let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
//     ajax_request("POST", "/cf_forecast/", data).then(function(response){
//         if (response !== null || response !== undefined){
//             $('#contentdata').html(response['fm_html']);
//             $('#forecast_graph').html(response['fm_model_html']);
//            // $('#mape_info').html(response['mape']);
//             document.getElementById("mape_info").innerText = response['mape'];
//             hideloading();
//         }
//     });
// });

$(document).on('click', '#forecast', function (event) {
    showloading();
    event.preventDefault();
    let project_id = $("#selectprojectbtn").val();
    let be_name = $("#select_billing").val();
    let data = {"project_id": project_id,"be_name":be_name, 'csrfmiddlewaretoken': $("#csrfmiddlewaretoken").val()};
    ajax_request("POST", "/cf_forecast/", data).then(function(response) {
        if (response !== null || response !== undefined) {
             if (response['best_model'] === "rf") {
                 //let best_mod = "Random Forest";
                 $('#contentdata').html(response['fm_html']);
                 $('#forecast_graph').html(response['fm_model_html']);
                 document.getElementById("mape_info").innerText = response['mape'];
                 document.getElementById("model_chosen").innerText = response['best_model'];
                 $('#feature_imp').html(response['rf_feature_imp_html']);
                 hideloading();
             }

             else {
                // let best_mod = "Arima";
                $('#contentdata').html(response['fm_html']);
                $('#forecast_graph').html(response['fm_model_html']);
                document.getElementById("mape_info").innerText = response['mape'];
                document.getElementById("model_chosen").innerText = response['best_model'];
                hideloading();
            }
    }
    });
});