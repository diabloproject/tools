;;-*- lexical-binding: t; -*-

(require 'url)
(require 'json)
(require 'evil nil t)

(setq discord-token "")
(setq discord-clonk-guild-id "1064660360071745686")
(setq discord-the-computer-channel "1064660360533135551")


(defun json-deserialize (s)
  (json-parse-string s :object-type 'plist))

(defun discord-get (uri)
  (let ((url-request-extra-headers `(("Authorization" . ,discord-token))))
    (with-current-buffer (url-retrieve-synchronously (concat "https://discord.com" uri) 'f)
      (goto-char (point-min))
      (re-search-forward "^$")
      (delete-region (point-min) (point))
      (let ((content (buffer-string)))
        (kill-buffer)
        (json-deserialize content)))))

(defun discord-my (entity)
  (discord-get (concat "/api/v10/users/@me/" entity)))

(defun discord-guild (guild-id)
  (discord-get(concat "/api/v10/guilds/" guild-id)))

(defun discord-guild-channels (guild-id)
  (discord-get(concat "/api/v10/guilds/" (format "%s" guild-id) "/channels")))

(defun discord-channel-messages (channel-id)
  (discord-get(concat "/api/v10/channels/" (format "%s" channel-id) "/messages")))

(defun discord-post (uri data)
  "POST DATA to URI."
  (let ((url-request-method "POST")
        (url-request-extra-headers `(("Authorization" . ,discord-token)
                                      ("Content-Type" . "application/json")))
        (url-request-data (json-serialize data)))
    (with-current-buffer (url-retrieve-synchronously (concat "https://discord.com" uri) 'f)
      (goto-char (point-min))
      (re-search-forward "^$")
      (delete-region (point-min) (point))
      (let ((content (buffer-string)))
        (kill-buffer)
        (json-deserialize content)))))

(defun discord-send-message (channel-id content)
  "Send a message with CONTENT to CHANNEL-ID."
  (discord-post (concat "/api/v10/channels/" (format "%s" channel-id) "/messages")
                `(:content ,content)))

;;; Discord Guild List Mode

(define-derived-mode discord-guild-list-mode tabulated-list-mode "Discord Guilds"
  "Major mode for Discord guild list."
  (setq tabulated-list-format [("Name" 40 t) ("ID" 20 t)])
  (setq tabulated-list-padding 2)
  (setq tabulated-list-sort-key (cons "Name" nil))
  (add-hook 'tabulated-list-revert-hook 'discord-guild-list--refresh nil t)
  (tabulated-list-init-header))

(define-key discord-guild-list-mode-map (kbd "RET") 'discord-guild-list-open-channels)
(define-key discord-guild-list-mode-map (kbd "q") 'quit-window)

(when (featurep 'evil)
  (evil-set-initial-state 'discord-guild-list-mode 'normal)
  (evil-define-key 'normal discord-guild-list-mode-map
    (kbd "RET") 'discord-guild-list-open-channels
    (kbd "q") 'quit-window
    (kbd "g r") 'revert-buffer))

(defun discord-guild-list--refresh ()
  "Refresh the guild list."
  (let ((guilds (append (discord-my "guilds") nil)))
    (setq tabulated-list-entries
          (mapcar (lambda (guild)
                    (list (plist-get guild :id)
                          (vector (plist-get guild :name)
                                  (plist-get guild :id))))
                  guilds))))

(defun discord-guild-list-open-channels ()
  "Open channel list for the guild at point."
  (interactive)
  (let ((guild-id (tabulated-list-get-id)))
    (when guild-id
      (discord-show-channels guild-id))))

(defun discord-show-guilds ()
  "Show Discord guild list."
  (interactive)
  (let ((buffer (get-buffer-create "*Discord Guilds*")))
    (with-current-buffer buffer
      (discord-guild-list-mode)
      (discord-guild-list--refresh)
      (tabulated-list-print t))
    (switch-to-buffer buffer)))

;;; Discord Channel List Mode

(defvar-local discord-current-guild-id nil
  "The current guild ID for this channel list buffer.")

(define-derived-mode discord-channel-list-mode tabulated-list-mode "Discord Channels"
  "Major mode for Discord channel list."
  (setq tabulated-list-format [("Type" 8 t) ("Name" 40 t) ("ID" 20 t)])
  (setq tabulated-list-padding 2)
  (setq tabulated-list-sort-key (cons "Name" nil))
  (tabulated-list-init-header))

(define-key discord-channel-list-mode-map (kbd "RET") 'discord-channel-list-open-messages)
(define-key discord-channel-list-mode-map (kbd "q") 'quit-window)
(define-key discord-channel-list-mode-map (kbd "g") 'discord-channel-list-refresh)

(when (featurep 'evil)
  (evil-set-initial-state 'discord-channel-list-mode 'normal)
  (evil-define-key 'normal discord-channel-list-mode-map
    (kbd "RET") 'discord-channel-list-open-messages
    (kbd "q") 'quit-window
    (kbd "g r") 'discord-channel-list-refresh))

(defun discord-channel-list-refresh ()
  "Refresh the channel list."
  (interactive)
  (when discord-current-guild-id
    (let ((channels (append (discord-guild-channels discord-current-guild-id) nil)))
      (setq tabulated-list-entries
            (mapcar (lambda (channel)
                      (list (plist-get channel :id)
                            (vector (number-to-string (plist-get channel :type))
                                    (plist-get channel :name)
                                    (plist-get channel :id))))
                    channels))
      (tabulated-list-print t))))

(defun discord-channel-list-open-messages ()
  "Open message list for the channel at point."
  (interactive)
  (let ((channel-id (tabulated-list-get-id)))
    (when channel-id
      (discord-show-messages channel-id))))

(defun discord-show-channels (guild-id)
  "Show Discord channel list for GUILD-ID."
  (interactive)
  (let ((buffer (get-buffer-create (format "*Discord Channels: %s*" guild-id))))
    (with-current-buffer buffer
      (discord-channel-list-mode)
      (setq discord-current-guild-id guild-id)
      (discord-channel-list-refresh))
    (switch-to-buffer buffer)))

;;; Discord Messages Mode

(defvar-local discord-current-channel-id nil
  "The current channel ID for this messages buffer.")

(define-derived-mode discord-messages-mode special-mode "Discord Messages"
  "Major mode for Discord messages."
  (setq buffer-read-only t))

(define-key discord-messages-mode-map (kbd "q") 'quit-window)
(define-key discord-messages-mode-map (kbd "g") 'discord-messages-refresh)
(define-key discord-messages-mode-map (kbd "i") 'discord-messages-send)

(when (featurep 'evil)
  (evil-set-initial-state 'discord-messages-mode 'normal)
  (evil-define-key 'normal discord-messages-mode-map
    (kbd "q") 'quit-window
    (kbd "g r") 'discord-messages-refresh
    (kbd "i") 'discord-messages-send))

(defun discord-format-timestamp (iso-timestamp)
  "Format ISO-TIMESTAMP to a readable format."
  (if (not iso-timestamp)
      ""
    (let* ((time (parse-iso8601-time-string iso-timestamp))
           (decoded (decode-time time)))
      (format-time-string "%Y-%m-%d %H:%M:%S" time))))

(defun discord-insert-image (url)
  "Insert image from URL inline."
  (let ((image-data (with-current-buffer
                        (url-retrieve-synchronously url 'silent)
                      (goto-char (point-min))
                      (re-search-forward "\n\n")
                      (delete-region (point-min) (point))
                      (buffer-string))))
    (when image-data
      (insert-image (create-image image-data nil t))
      (insert "\n"))))

(defun discord-messages-refresh ()
  "Refresh the messages."
  (interactive)
  (when discord-current-channel-id
    (let ((messages (append (discord-channel-messages discord-current-channel-id) nil))
          (inhibit-read-only t))
      (erase-buffer)
      (goto-char (point-min))
      (insert (format "=== Channel: %s ===\n\n" discord-current-channel-id))
      (dolist (msg (reverse messages))
        (let ((author (plist-get msg :author))
              (content (plist-get msg :content))
              (timestamp (plist-get msg :timestamp))
              (attachments (plist-get msg :attachments)))
          (insert (format "[%s] %s: %s"
                          (discord-format-timestamp timestamp)
                          (plist-get author :username)
                          content))
          (when (and attachments (> (length attachments) 0))
            (insert "\n")
            (dolist (attachment (append attachments nil))
              (let ((url (plist-get attachment :url))
                    (content-type (plist-get attachment :content_type)))
                (when (and content-type (string-match-p "^image/" content-type))
                  (condition-case nil
                      (discord-insert-image url)
                    (error (insert (format "[Image: %s]\n" url))))))))
          (insert "\n")))
      (goto-char (point-min)))))

(defun discord-show-messages (channel-id)
  "Show Discord messages for CHANNEL-ID."
  (interactive)
  (let ((buffer (get-buffer-create (format "*Discord Messages: %s*" channel-id))))
    (with-current-buffer buffer
      (discord-messages-mode)
      (setq discord-current-channel-id channel-id)
      (discord-messages-refresh))
    (switch-to-buffer buffer)))

(defun discord-messages-send ()
  "Send a message to the current channel."
  (interactive)
  (when discord-current-channel-id
    (let ((content (read-string "Message: ")))
      (when (and content (> (length content) 0))
        (discord-send-message discord-current-channel-id content)
        (discord-messages-refresh)))))

(provide 'main)
;;; main.el ends here
