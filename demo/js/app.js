var username = ''


// send the message to user
function send_message(message){
      var prevSms = $('.container').html();
      if (prevSms.length > 8) {
        prevSms = prevSms + '<br>'
        }
      $('.container').html(prevSms + '<span class="cureent_sms">' + '<span class="bot">Chatbot: </span>' + message + '</span>');

      $('.cureent_sms').hide();
      $('.cureent_sms').delay(50).fadeIn();
      $('.cureent_sms').removeClass("current_sms");
    }

// get the username
function get_username(){
    send_message('Hi, friend what is your name....?');
}

// simple ai function
function ai(message){
        if (username.length < 3){
          username = message;
          send_message('Hi, nice to meet you ' + username + ', how are you doing.. ')
        }

        if ((message.indexOf('how are you') >= 0) || (message.indexOf('about you') >= 0)){
          send_message('Am okey, thanks for ask ' + '<i>' + username + '</i>');
        }

        if ((message.indexOf('what is your name') >= 0) || (message.indexOf('name') >= 0)){
          send_message('My name in ChatBox.. am here to chat with you..');
        }

        if ((message.indexOf('old') >= 0) || (message.indexOf('age') >= 0)){
          send_message('I do not know how old i am.. am sorry..!!');
        }

        if ((message.indexOf('sex') >= 0) || (message.indexOf('love') >= 0)){
          send_message('Am sorry i can not tell you about that.');
        }

        if (message.indexOf('time') >= 0){
          var date = new Date();
          var hour = date.getHours();
          var minutes = date.getMinutes();
          send_message('Current time is: ' + hour + ':' + minutes );
        }
}

// main JQuery function
$(function() {
      // this function is used to call username of user;
      get_username();

      $('#text-sms').keypress(function(event){
        if (event.which == 13) {
          if ($('#enter').prop('checked')){
            $('#send-button').click();
            event.preventDefault();
          }
        }
      });

    $('#send-button').click(function(){
        var username = '<span class="username">You: </span>'
        var sms = $('#text-sms').val();
        $('#text-sms').val('');
          //store the first sms
        var prevSms = $('.container').html();

          //check if prev sms length is greater than 8
        if (prevSms.length > 8) {
          prevSms = prevSms + '<br>'
          }

        //show the sms to the container div
        $('.container').html(prevSms + username + sms);

        $('.container').scrollTop($('.container').prop('scrollHeight'));
        // the main function
        ai(sms);
      });
});
