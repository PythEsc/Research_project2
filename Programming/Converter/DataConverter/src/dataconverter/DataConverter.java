/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dataconverter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Tobi
 */
public class DataConverter
{
  public static final String PREFIX_TESCO = "TE_";
  public static final String PREFIX_SAINSBURY = "SA_";
  public static final String[] IGNORE_LIST =
  {
    "January", "February"
  };
  public static final String PREFIX_FILE = "page";
  public static final String POSTFIX_FILE_COMMENT = "comments.tab";
  public static final String POSTFIX_FILE_FULLSTAT = "fullstats.tab";
  private static final char DEFAULT_SEPARATOR = ';';

  private static String outputFolderTesco = "Data/Tesco";
  private static String outputFolderSainsbury = "Data/Sainsbury";

  private static String firstLine_comments =
      "position	post_id	post_by	post_text	post_published	comment_id	comment_by	is_reply	comment_message	comment_published	comment_like_count";
  private static String firstLine_fullstats =
      "type	by	post_id	post_link	post_message	picture	full_picture	link	link_domain	post_published	post_published_unix	post_published_sql	likes_count_fb	comments_count_fb	reactions_count_fb	shares_count_fb	engagement_fb	comments_retrieved	comments_base	comments_replies	comment_likes_count	rea_NONE	rea_LIKE	rea_LOVE	rea_WOW	rea_HAHA	rea_SAD	rea_ANGRY	rea_THANKFUL";

  /**
   * @param args the command line arguments
   */
  public static void main(String[] args)
  {

    Runnable runnableSains = () ->
    {
      String sainsburyPath = "Sainsbury_s";
      File sainsbury = new File(sainsburyPath);
      File outputSainsbury = new File(outputFolderSainsbury);
      if(!outputSainsbury.exists())
      {
        outputSainsbury.mkdir();
      }
      File[] listFiles = sainsbury.listFiles();
      for(File listFile : listFiles)
      {
        String name = listFile.getName();
        if(name.startsWith(PREFIX_SAINSBURY)
            && !name.endsWith(IGNORE_LIST[0])
            && !name.endsWith(IGNORE_LIST[1]))
        {
          convertFolder(listFile, outputFolderSainsbury);
        }
      }
    };

    Runnable runnableTesco = () ->
    {
      String tescozipPath = "Tesco";
      File tesco = new File(tescozipPath);
      File outputTesco = new File(outputFolderTesco);
      if(!outputTesco.exists())
      {
        outputTesco.mkdir();
      }
      File[] listFiles = tesco.listFiles();
      for(File listFile : listFiles)
      {
        String name = listFile.getName();
        if(name.startsWith(PREFIX_TESCO)
            && !name.endsWith(IGNORE_LIST[0])
            && !name.endsWith(IGNORE_LIST[1]))
        {
          convertFolder(listFile, outputFolderTesco);
        }
      }
    };
    Thread sains = new Thread(runnableSains);
    Thread tesc = new Thread(runnableTesco);

    sains.start();

    tesc.start();

  }

  /**
   *
   *
   * @param file
   * @param outputFolder
   */
  public static void convertFolder(File file, String outputFolder)
  {
    ArrayList<String> mapFullstatsIds = new ArrayList<>();
    File commentFile = null;
    File fullStatsFile = null;
    for(File listFile : file.listFiles())
    {
      if(listFile.getName().endsWith(POSTFIX_FILE_COMMENT))
      {
        commentFile = listFile;
      }
      else if(listFile.getName().endsWith(POSTFIX_FILE_FULLSTAT))
      {
        fullStatsFile = listFile;
      }
    }

    File parent = new File(outputFolder + "/" + file.getName());
    if(!parent.exists())
    {
      parent.mkdir();
    }
    //Compute
    File fullstats = new File(outputFolder + "/" + file.getName()
        + "/fullstats.csv");

    checkValidIdsAndWriteLinesFullstats(fullStatsFile, fullstats, mapFullstatsIds);
    writeLinesComments(outputFolder, file, commentFile, mapFullstatsIds);

  }

  private static void writeLinesComments(String outputFolder, File file,
      File commentFile, ArrayList<String> mapFullstatsIds)
  {
    try
    {
      File comments = new File(outputFolder + "/" + file.getName()
          + "/comments.csv");
      Writer commentsFW = new BufferedWriter(new OutputStreamWriter(
          new FileOutputStream(comments), "UTF-8"));
      try(Scanner sc = new Scanner(commentFile, "UTF-8"))
      {
        sc.useLocale(new Locale("gb"));
        boolean firstLineFound = false;
        while(sc.hasNextLine())
        {
          String nextLine = sc.nextLine();
          if(!firstLineFound
              && nextLine.equals(firstLine_comments))
          {
            firstLineFound = true;
          }
          else if(firstLineFound)
          {
            String[] split = nextLine.split("\t");
            String id = split[1];
            ArrayList<String> listToAdd = new ArrayList<>();
            listToAdd.addAll(Arrays.asList(split));
            if(mapFullstatsIds.contains(id) && !listToAdd.isEmpty())
            {
              writeLine(commentsFW, listToAdd);
            }
          }
        }
      }
    }
    catch(IOException ex)
    {
      Logger.getLogger(DataConverter.class.getName()).
          log(Level.SEVERE, null, ex);
    }
  }

  private static void checkValidIdsAndWriteLinesFullstats(File fullStatsFile, File fullstats,
      ArrayList<String> mapFullstatsIds)
  {
    try
    {
      try(Scanner sc = new Scanner(fullStatsFile, "UTF-8"))
      {
        Writer fullstatsFw = new BufferedWriter(new OutputStreamWriter(
            new FileOutputStream(fullstats), "UTF-8"));
        sc.useLocale(new Locale("gb"));
        boolean firstLineFound = false;
        while(sc.hasNextLine())
        {
          String nextLine = sc.nextLine();
          if(!firstLineFound
              && nextLine.equals(firstLine_fullstats))
          {
            firstLineFound = true;
          }
          else if(firstLineFound)
          {
            String[] split = nextLine.split("\t");
            String text = split[4];
            if(text.trim().length() > 20
                && split[5].trim().isEmpty())
            {
              String id = split[2];
              mapFullstatsIds.add(id);
              ArrayList<String> listToAdd = new ArrayList<>();
              listToAdd.addAll(Arrays.asList(split));
              if(!listToAdd.isEmpty())
              {
                writeLine(fullstatsFw, listToAdd);
              }
            }
          }
        }
      }
    }
    catch(IOException ex)
    {
      Logger.getLogger(DataConverter.class.getName()).
          log(Level.SEVERE, null, ex);
    }
  }

  public static void writeLine(Writer w, List<String> values) throws IOException
  {
    writeLine(w, values, DEFAULT_SEPARATOR, ' ');
  }

  private static String followCVSformat(String value)
  {

    String result = value;
    if(result.contains("\""))
    {
      result = result.replace("\"", "\"\"");
    }
    if(result.contains(";"))
    {
      result = "\"" + result + "\"";
    }
    return result;

  }

  public static void writeLine(Writer w, List<String> values, char separators,
      char customQuote) throws IOException
  {

    boolean first = true;

    //default customQuote is empty
    if(separators == ' ')
    {
      separators = DEFAULT_SEPARATOR;
    }

    StringBuilder sb = new StringBuilder();
    for(String value : values)
    {
      if(!first)
      {
        sb.append(separators);
      }
      if(customQuote == ' ')
      {
        sb.append(followCVSformat(value));
      }
      else
      {
        sb.append(customQuote).append(followCVSformat(value)).
            append(customQuote);
      }

      first = false;
    }
    sb.append("\n");
    w.append(sb.toString());

  }

}
